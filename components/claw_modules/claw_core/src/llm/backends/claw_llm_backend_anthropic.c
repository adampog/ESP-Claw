/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "llm/backends/claw_llm_backend_anthropic.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "esp_log.h"
#include "llm/claw_llm_http_transport.h"
#include "llm/media/claw_media_pipeline.h"

static const char *TAG = "llm_anthropic";

#define CLAW_LLM_ANTHROPIC_VERSION "2023-06-01"
/*
 * Current Claude models always think, and thinking counts toward max_tokens.
 * 16K leaves room for thinking plus the reply while keeping non-streaming
 * requests within typical HTTP timeouts.
 */
#define CLAW_LLM_ANTHROPIC_MAX_TOKENS 16000
#define CLAW_LLM_ANTHROPIC_OFFICIAL_HOST "https://api.anthropic.com"
/* Server-side refusal fallback, "default" form (routes by refusal category). */
#define CLAW_LLM_ANTHROPIC_FALLBACK_BETA "server-side-fallback-2026-07-01"
#define CLAW_LLM_ANTHROPIC_MAX_PAUSE_CONTINUATIONS 5
#define CLAW_LLM_ANTHROPIC_CONTENT_FORMAT "anthropic"
#define CLAW_LLM_ANTHROPIC_PREFIX_HEX_LEN 16

typedef struct {
    char *api_key;
    char *model;
    char *base_url;
    char *effort;
    uint32_t timeout_ms;
    size_t image_max_bytes;
    /* base_url points at the Anthropic API itself, not a proxy or gateway. */
    bool official_api;
    /* Send fallbacks:"default" (official API + a model that supports it). */
    bool server_fallback;
    /* Sequence number given to the next assistant turn we hand back. */
    uint32_t next_seq;
    /* Thinking blocks of turns with seq < strip_before_seq are never replayed. */
    uint32_t strip_before_seq;
} anthropic_backend_ctx_t;

/* Models that accept `fallbacks: "default"` on the Claude API. */
static const char *const s_server_fallback_models[] = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
    "claude-fable-5",
};

/* Models that reject output_config.effort. */
static const char *const s_no_effort_model_prefixes[] = {
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
};

static char *dup_printf(const char *fmt, ...)
{
    va_list args;
    va_list copy;
    int needed;
    char *buf;

    va_start(args, fmt);
    va_copy(copy, args);
    needed = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (needed < 0) {
        va_end(args);
        return NULL;
    }

    buf = calloc(1, (size_t)needed + 1);
    if (!buf) {
        va_end(args);
        return NULL;
    }

    vsnprintf(buf, (size_t)needed + 1, fmt, args);
    va_end(args);
    return buf;
}

static char *dup_n_string(const char *value, size_t len)
{
    char *copy = NULL;

    if (!value) {
        return NULL;
    }

    copy = calloc(1, len + 1);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, value, len);
    copy[len] = '\0';
    return copy;
}

static char *join_url(const char *base_url, const char *path)
{
    bool base_has_slash;
    bool path_has_slash;

    if (!base_url || !path) {
        return NULL;
    }

    base_has_slash = base_url[0] && base_url[strlen(base_url) - 1] == '/';
    path_has_slash = path[0] == '/';
    if (base_has_slash && path_has_slash) {
        return dup_printf("%s%s", base_url, path + 1);
    }
    if (!base_has_slash && !path_has_slash) {
        return dup_printf("%s/%s", base_url, path);
    }
    return dup_printf("%s%s", base_url, path);
}

static esp_err_t dup_json_string(cJSON *json, char **out_value)
{
    if (!json || !cJSON_IsString(json) || !json->valuestring || !out_value) {
        return ESP_ERR_INVALID_ARG;
    }

    *out_value = strdup(json->valuestring);
    return *out_value ? ESP_OK : ESP_ERR_NO_MEM;
}

static bool json_type_is(const cJSON *block, const char *type)
{
    const cJSON *type_json = NULL;

    if (!block || !cJSON_IsObject(block) || !type) {
        return false;
    }
    type_json = cJSON_GetObjectItemCaseSensitive(block, "type");
    return cJSON_IsString(type_json) && type_json->valuestring &&
           strcmp(type_json->valuestring, type) == 0;
}

static bool block_is_thinking(const cJSON *block)
{
    return json_type_is(block, "thinking") || json_type_is(block, "redacted_thinking");
}

static bool string_in_list(const char *value, const char *const *list, size_t count, bool prefix)
{
    size_t i;

    if (!value) {
        return false;
    }
    for (i = 0; i < count; i++) {
        if (prefix ? strncmp(value, list[i], strlen(list[i])) == 0 : strcmp(value, list[i]) == 0) {
            return true;
        }
    }
    return false;
}

static bool is_official_api_url(const char *base_url)
{
    size_t host_len = strlen(CLAW_LLM_ANTHROPIC_OFFICIAL_HOST);
    size_t len;

    if (!base_url) {
        return false;
    }
    len = strlen(base_url);
    while (len > 0 && base_url[len - 1] == '/') {
        len--;
    }
    if (len < host_len || strncasecmp(base_url, CLAW_LLM_ANTHROPIC_OFFICIAL_HOST, host_len) != 0) {
        return false;
    }
    return len == host_len ||
           (len == host_len + 3 && strncmp(base_url + host_len, "/v1", 3) == 0);
}

static bool is_valid_effort(const char *effort)
{
    static const char *const levels[] = { "low", "medium", "high", "xhigh", "max" };

    return string_in_list(effort, levels, sizeof(levels) / sizeof(levels[0]), false);
}

/* ---------- Prefix fingerprint (preserved thinking) ---------- */

/*
 * A thinking block is only valid while everything it was produced after is
 * unchanged: the top-level system prompt, the tools, and every earlier
 * message. We fingerprint that part of the request (system, tools, turn
 * context and the messages up to the current user turn; later messages are
 * append-only tool-loop turns) and store the fingerprint with each assistant
 * turn. Thinking is replayed only for turns whose fingerprint still matches;
 * otherwise it is stripped before sending (the documented recovery), so a
 * prompt change between tool steps (a skill or memory update) degrades to
 * "no earlier reasoning" instead of a 400.
 */
static void fnv1a_update(uint64_t *hash, const void *data, size_t len)
{
    const uint8_t *bytes = (const uint8_t *)data;
    size_t i;

    for (i = 0; i < len; i++) {
        *hash ^= bytes[i];
        *hash *= 0x100000001b3ULL;
    }
}

static void fnv1a_update_field(uint64_t *hash, const char *value)
{
    static const uint8_t separator = 0x1f;

    if (value) {
        fnv1a_update(hash, value, strlen(value));
    }
    fnv1a_update(hash, &separator, 1);
}

static int find_last_user_message(cJSON *messages)
{
    cJSON *msg = NULL;
    int index = 0;
    int last = -1;

    cJSON_ArrayForEach(msg, messages) {
        cJSON *role = cJSON_GetObjectItemCaseSensitive(msg, "role");

        if (cJSON_IsString(role) && role->valuestring && strcmp(role->valuestring, "user") == 0) {
            last = index;
        }
        index++;
    }
    return last;
}

static esp_err_t compute_prefix_fingerprint(const claw_llm_chat_request_t *request,
                                            int last_user_index,
                                            char out_hex[CLAW_LLM_ANTHROPIC_PREFIX_HEX_LEN + 1])
{
    uint64_t hash = 0xcbf29ce484222325ULL;
    cJSON *msg = NULL;
    int index = 0;

    fnv1a_update_field(&hash, request->system_prompt);
    fnv1a_update_field(&hash, request->tools_json);
    fnv1a_update_field(&hash, request->turn_context);
    cJSON_ArrayForEach(msg, request->messages) {
        char *printed = NULL;

        if (index > last_user_index) {
            break;
        }
        printed = cJSON_PrintUnformatted(msg);
        if (!printed) {
            return ESP_ERR_NO_MEM;
        }
        fnv1a_update_field(&hash, printed);
        free(printed);
        index++;
    }

    snprintf(out_hex, CLAW_LLM_ANTHROPIC_PREFIX_HEX_LEN + 1, "%016" PRIx64, hash);
    return ESP_OK;
}

/* ---------- Request conversion ---------- */

static cJSON *anthropic_make_text_block(const char *text)
{
    cJSON *block = NULL;

    if (!text || !text[0]) {
        return NULL;
    }

    block = cJSON_CreateObject();
    if (!block) {
        return NULL;
    }
    cJSON_AddStringToObject(block, "type", "text");
    cJSON_AddStringToObject(block, "text", text);
    return block;
}

static cJSON *anthropic_make_cache_control(void)
{
    cJSON *cache_control = cJSON_CreateObject();

    if (cache_control) {
        cJSON_AddStringToObject(cache_control, "type", "ephemeral");
    }
    return cache_control;
}

static cJSON *anthropic_make_tool_use_block(cJSON *tool_call)
{
    cJSON *block = NULL;
    cJSON *id_json = NULL;
    cJSON *function_json = NULL;
    cJSON *name_json = NULL;
    cJSON *args_json = NULL;
    cJSON *input_json = NULL;

    if (!tool_call || !cJSON_IsObject(tool_call)) {
        return NULL;
    }

    id_json = cJSON_GetObjectItem(tool_call, "id");
    function_json = cJSON_GetObjectItem(tool_call, "function");
    name_json = cJSON_IsObject(function_json) ? cJSON_GetObjectItem(function_json, "name") : NULL;
    args_json = cJSON_IsObject(function_json) ? cJSON_GetObjectItem(function_json, "arguments") : NULL;
    if (!cJSON_IsString(id_json) || !cJSON_IsString(name_json)) {
        return NULL;
    }

    block = cJSON_CreateObject();
    if (!block) {
        return NULL;
    }
    cJSON_AddStringToObject(block, "type", "tool_use");
    cJSON_AddStringToObject(block, "id", id_json->valuestring);
    cJSON_AddStringToObject(block, "name", name_json->valuestring);

    if (cJSON_IsString(args_json) && args_json->valuestring && args_json->valuestring[0]) {
        input_json = cJSON_Parse(args_json->valuestring);
    }
    if (!input_json) {
        input_json = cJSON_CreateObject();
    }
    if (!input_json) {
        cJSON_Delete(block);
        return NULL;
    }
    cJSON_AddItemToObject(block, "input", input_json);
    return block;
}

static cJSON *anthropic_make_tool_result_block(const char *tool_call_id, const char *content)
{
    cJSON *block = NULL;

    if (!tool_call_id || !tool_call_id[0]) {
        return NULL;
    }

    block = cJSON_CreateObject();
    if (!block) {
        return NULL;
    }
    cJSON_AddStringToObject(block, "type", "tool_result");
    cJSON_AddStringToObject(block, "tool_use_id", tool_call_id);
    cJSON_AddStringToObject(block, "content", content ? content : "");
    cJSON_AddBoolToObject(block, "is_error", false);
    return block;
}

static cJSON *anthropic_duplicate_supported_block(cJSON *block)
{
    if (json_type_is(block, "text") ||
            json_type_is(block, "tool_use") ||
            json_type_is(block, "tool_result")) {
        return cJSON_Duplicate(block, true);
    }
    return NULL;
}

static cJSON *provider_content_blocks(cJSON *msg)
{
    cJSON *provider_content = cJSON_GetObjectItemCaseSensitive(msg, CLAW_LLM_PROVIDER_CONTENT_KEY);
    cJSON *format = NULL;
    cJSON *content = NULL;

    if (!cJSON_IsObject(provider_content)) {
        return NULL;
    }
    format = cJSON_GetObjectItemCaseSensitive(provider_content, "format");
    content = cJSON_GetObjectItemCaseSensitive(provider_content, "content");
    if (!cJSON_IsString(format) || !format->valuestring ||
            strcmp(format->valuestring, CLAW_LLM_ANTHROPIC_CONTENT_FORMAT) != 0 ||
            !cJSON_IsArray(content)) {
        return NULL;
    }
    return content;
}

static bool provider_content_keeps_thinking(const anthropic_backend_ctx_t *ctx,
                                            cJSON *msg,
                                            const char *prefix_hex,
                                            bool strip_all_thinking)
{
    cJSON *provider_content = cJSON_GetObjectItemCaseSensitive(msg, CLAW_LLM_PROVIDER_CONTENT_KEY);
    cJSON *prefix = cJSON_GetObjectItemCaseSensitive(provider_content, "prefix");
    cJSON *seq = cJSON_GetObjectItemCaseSensitive(provider_content, "seq");

    if (strip_all_thinking || !prefix_hex) {
        return false;
    }
    if (!cJSON_IsString(prefix) || !prefix->valuestring || strcmp(prefix->valuestring, prefix_hex) != 0) {
        return false;
    }
    return cJSON_IsNumber(seq) && seq->valuedouble >= (double)ctx->strip_before_seq;
}

/*
 * Converts the OpenAI-shaped claw history to Anthropic messages.
 * - Assistant turns that carry provider_content are replayed verbatim
 *   (thinking blocks included while the prefix fingerprint still matches).
 * - Consecutive "tool" messages become one user message with all
 *   tool_result blocks, as the API expects for parallel tool calls.
 * - turn_context becomes the first text block of the current user turn.
 */
static cJSON *convert_messages_to_anthropic(const anthropic_backend_ctx_t *ctx,
                                            cJSON *messages,
                                            const char *turn_context,
                                            int last_user_index,
                                            const char *prefix_hex,
                                            bool strip_all_thinking,
                                            bool *out_replays_thinking)
{
    cJSON *out = NULL;
    cJSON *msg = NULL;
    cJSON *tool_result_blocks = NULL;
    int index = -1;

    if (out_replays_thinking) {
        *out_replays_thinking = false;
    }
    out = cJSON_CreateArray();
    if (!out) {
        return NULL;
    }
    if (!messages || !cJSON_IsArray(messages)) {
        return out;
    }

    cJSON_ArrayForEach(msg, messages) {
        cJSON *role_json = cJSON_GetObjectItem(msg, "role");
        cJSON *content_json = cJSON_GetObjectItem(msg, "content");
        const char *role = cJSON_IsString(role_json) ? role_json->valuestring : NULL;
        cJSON *out_message = NULL;
        cJSON *blocks = NULL;
        cJSON *native_blocks = NULL;

        index++;
        if (!role || !role[0]) {
            continue;
        }

        if (strcmp(role, "tool") == 0) {
            cJSON *tool_call_id = cJSON_GetObjectItem(msg, "tool_call_id");
            cJSON *tool_block = anthropic_make_tool_result_block(
                                    cJSON_IsString(tool_call_id) ? tool_call_id->valuestring : NULL,
                                    cJSON_IsString(content_json) ? content_json->valuestring : "");

            if (!tool_block) {
                cJSON_Delete(out);
                return NULL;
            }
            if (!tool_result_blocks) {
                cJSON *tool_message = cJSON_CreateObject();

                tool_result_blocks = cJSON_CreateArray();
                if (!tool_message || !tool_result_blocks) {
                    cJSON_Delete(tool_message);
                    cJSON_Delete(tool_result_blocks);
                    cJSON_Delete(tool_block);
                    cJSON_Delete(out);
                    return NULL;
                }
                cJSON_AddStringToObject(tool_message, "role", "user");
                cJSON_AddItemToObject(tool_message, "content", tool_result_blocks);
                cJSON_AddItemToArray(out, tool_message);
            }
            cJSON_AddItemToArray(tool_result_blocks, tool_block);
            continue;
        }

        if (strcmp(role, "assistant") != 0 && strcmp(role, "user") != 0) {
            continue;
        }
        tool_result_blocks = NULL;

        out_message = cJSON_CreateObject();
        blocks = cJSON_CreateArray();
        if (!out_message || !blocks) {
            cJSON_Delete(out_message);
            cJSON_Delete(blocks);
            cJSON_Delete(out);
            return NULL;
        }

        cJSON_AddStringToObject(out_message, "role", role);

        if (strcmp(role, "user") == 0 && index == last_user_index &&
                turn_context && turn_context[0]) {
            cJSON *context_block = anthropic_make_text_block(turn_context);

            if (!context_block) {
                cJSON_Delete(out_message);
                cJSON_Delete(blocks);
                cJSON_Delete(out);
                return NULL;
            }
            cJSON_AddItemToArray(blocks, context_block);
        }

        native_blocks = strcmp(role, "assistant") == 0 ? provider_content_blocks(msg) : NULL;
        if (native_blocks) {
            bool keep_thinking = provider_content_keeps_thinking(ctx, msg, prefix_hex, strip_all_thinking);
            cJSON *block = NULL;

            cJSON_ArrayForEach(block, native_blocks) {
                cJSON *dup = NULL;

                if (block_is_thinking(block)) {
                    if (!keep_thinking) {
                        continue;
                    }
                    if (out_replays_thinking) {
                        *out_replays_thinking = true;
                    }
                }
                dup = cJSON_Duplicate(block, true);
                if (!dup) {
                    cJSON_Delete(out_message);
                    cJSON_Delete(blocks);
                    cJSON_Delete(out);
                    return NULL;
                }
                cJSON_AddItemToArray(blocks, dup);
            }
        } else {
            if (cJSON_IsString(content_json) && content_json->valuestring && content_json->valuestring[0]) {
                cJSON *text_block = anthropic_make_text_block(content_json->valuestring);

                if (!text_block) {
                    cJSON_Delete(out_message);
                    cJSON_Delete(blocks);
                    cJSON_Delete(out);
                    return NULL;
                }
                cJSON_AddItemToArray(blocks, text_block);
            } else if (cJSON_IsArray(content_json)) {
                cJSON *block = NULL;

                cJSON_ArrayForEach(block, content_json) {
                    cJSON *dup = anthropic_duplicate_supported_block(block);

                    if (dup) {
                        cJSON_AddItemToArray(blocks, dup);
                    }
                }
            }

            if (strcmp(role, "assistant") == 0) {
                cJSON *tool_calls = cJSON_GetObjectItem(msg, "tool_calls");
                cJSON *tool_call = NULL;

                if (cJSON_IsArray(tool_calls)) {
                    cJSON_ArrayForEach(tool_call, tool_calls) {
                        cJSON *tool_block = anthropic_make_tool_use_block(tool_call);

                        if (!tool_block) {
                            cJSON_Delete(out_message);
                            cJSON_Delete(blocks);
                            cJSON_Delete(out);
                            return NULL;
                        }
                        cJSON_AddItemToArray(blocks, tool_block);
                    }
                }
            }
        }

        if (cJSON_GetArraySize(blocks) == 0) {
            cJSON_Delete(out_message);
            cJSON_Delete(blocks);
            continue;
        }

        cJSON_AddItemToObject(out_message, "content", blocks);
        cJSON_AddItemToArray(out, out_message);
    }

    return out;
}

static cJSON *convert_tools_to_anthropic(const char *tools_json)
{
    cJSON *parsed = NULL;
    cJSON *out = NULL;
    cJSON *item = NULL;

    if (!tools_json || !tools_json[0]) {
        return NULL;
    }

    parsed = cJSON_Parse(tools_json);
    if (!parsed || !cJSON_IsArray(parsed)) {
        cJSON_Delete(parsed);
        return NULL;
    }

    out = cJSON_CreateArray();
    if (!out) {
        cJSON_Delete(parsed);
        return NULL;
    }

    cJSON_ArrayForEach(item, parsed) {
        cJSON *tool = NULL;
        cJSON *function_json = NULL;
        cJSON *name_json = NULL;
        cJSON *desc_json = NULL;
        cJSON *schema_json = NULL;

        if (cJSON_IsObject(item)) {
            cJSON *type_json = cJSON_GetObjectItem(item, "type");

            if (cJSON_IsString(type_json) && type_json->valuestring &&
                    strcmp(type_json->valuestring, "function") == 0) {
                function_json = cJSON_GetObjectItem(item, "function");
                name_json = cJSON_IsObject(function_json) ? cJSON_GetObjectItem(function_json, "name") : NULL;
                desc_json = cJSON_IsObject(function_json) ? cJSON_GetObjectItem(function_json, "description") : NULL;
                schema_json = cJSON_IsObject(function_json) ? cJSON_GetObjectItem(function_json, "parameters") : NULL;
            } else {
                name_json = cJSON_GetObjectItem(item, "name");
                desc_json = cJSON_GetObjectItem(item, "description");
                schema_json = cJSON_GetObjectItem(item, "input_schema");
            }
        }

        if (!cJSON_IsString(name_json) || !name_json->valuestring || !name_json->valuestring[0]) {
            continue;
        }

        tool = cJSON_CreateObject();
        if (!tool) {
            cJSON_Delete(parsed);
            cJSON_Delete(out);
            return NULL;
        }

        cJSON_AddStringToObject(tool, "name", name_json->valuestring);
        if (cJSON_IsString(desc_json) && desc_json->valuestring) {
            cJSON_AddStringToObject(tool, "description", desc_json->valuestring);
        }
        if (schema_json) {
            cJSON_AddItemToObject(tool, "input_schema", cJSON_Duplicate(schema_json, true));
        } else {
            cJSON_AddItemToObject(tool, "input_schema", cJSON_CreateObject());
        }
        cJSON_AddItemToArray(out, tool);
    }

    cJSON_Delete(parsed);
    return out;
}

static esp_err_t parse_data_url(const char *data_url, char **out_mime, char **out_data)
{
    const char *prefix = "data:";
    const char *mime_end = NULL;
    const char *data_start = NULL;

    if (out_mime) {
        *out_mime = NULL;
    }
    if (out_data) {
        *out_data = NULL;
    }
    if (!data_url || strncmp(data_url, prefix, strlen(prefix)) != 0 || !out_mime || !out_data) {
        return ESP_ERR_INVALID_ARG;
    }

    mime_end = strstr(data_url, ";base64,");
    if (!mime_end) {
        return ESP_ERR_INVALID_RESPONSE;
    }
    data_start = mime_end + strlen(";base64,");
    if (!data_start[0]) {
        return ESP_ERR_INVALID_RESPONSE;
    }

    *out_mime = dup_n_string(data_url + strlen(prefix), (size_t)(mime_end - (data_url + strlen(prefix))));
    *out_data = strdup(data_start);
    if (!*out_mime || !*out_data) {
        free(*out_mime);
        free(*out_data);
        *out_mime = NULL;
        *out_data = NULL;
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}

/* Fields shared by chat and media requests: effort and refusal fallback. */
static void add_common_request_fields(const anthropic_backend_ctx_t *ctx, cJSON *body)
{
    if (ctx->effort) {
        cJSON *output_config = cJSON_AddObjectToObject(body, "output_config");

        if (output_config) {
            cJSON_AddStringToObject(output_config, "effort", ctx->effort);
        }
    }
    if (ctx->server_fallback) {
        cJSON_AddStringToObject(body, "fallbacks", "default");
    }
}

/* ---------- Response handling ---------- */

static int json_int(const cJSON *object, const char *key)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(object, key);

    return cJSON_IsNumber(item) ? item->valueint : 0;
}

static void log_usage(const anthropic_backend_ctx_t *ctx, cJSON *root, const char *stop_reason)
{
    cJSON *usage = cJSON_GetObjectItemCaseSensitive(root, "usage");
    cJSON *model = cJSON_GetObjectItemCaseSensitive(root, "model");

    ESP_LOGI(TAG,
             "model=%s stop=%s in=%d cache_read=%d cache_write=%d out=%d",
             cJSON_IsString(model) && model->valuestring ? model->valuestring : ctx->model,
             stop_reason ? stop_reason : "(none)",
             json_int(usage, "input_tokens"),
             json_int(usage, "cache_read_input_tokens"),
             json_int(usage, "cache_creation_input_tokens"),
             json_int(usage, "output_tokens"));
}

static char *build_refusal_message(cJSON *root)
{
    cJSON *details = cJSON_GetObjectItemCaseSensitive(root, "stop_details");
    cJSON *category = cJSON_GetObjectItemCaseSensitive(details, "category");
    cJSON *explanation = cJSON_GetObjectItemCaseSensitive(details, "explanation");
    bool has_explanation = cJSON_IsString(explanation) && explanation->valuestring &&
                           explanation->valuestring[0];

    return dup_printf("Claude declined this request (stop_reason=refusal, category=%s)%s%s",
                      cJSON_IsString(category) && category->valuestring ? category->valuestring : "unspecified",
                      has_explanation ? ": " : "",
                      has_explanation ? explanation->valuestring : "");
}

/*
 * After a server-side fallback the content can contain `fallback` marker
 * blocks. Blocks before the last marker that are not text (thinking,
 * tool_use, unpaired server-tool blocks) must not be echoed back, and their
 * tool calls must not run. ESP-Claw declares no server tools, so only text
 * survives before the marker; the markers themselves are dropped.
 */
static cJSON *select_replay_blocks(cJSON *content)
{
    cJSON *kept = NULL;
    cJSON *block = NULL;
    int last_fallback = -1;
    int index = 0;

    cJSON_ArrayForEach(block, content) {
        if (json_type_is(block, "fallback")) {
            last_fallback = index;
        }
        index++;
    }

    kept = cJSON_CreateArray();
    if (!kept) {
        return NULL;
    }
    index = 0;
    cJSON_ArrayForEach(block, content) {
        bool keep = !json_type_is(block, "fallback") &&
                    (index > last_fallback || json_type_is(block, "text"));

        index++;
        if (!keep) {
            continue;
        }
        cJSON *dup = cJSON_Duplicate(block, true);
        if (!dup) {
            cJSON_Delete(kept);
            return NULL;
        }
        cJSON_AddItemToArray(kept, dup);
    }
    return kept;
}

static esp_err_t fill_response_from_content(anthropic_backend_ctx_t *ctx,
                                            cJSON *content,
                                            const char *stop_reason,
                                            const char *prefix_hex,
                                            claw_llm_response_t *out_response,
                                            char **out_error_message)
{
    cJSON *kept = NULL;
    cJSON *block = NULL;
    size_t tool_count = 0;
    size_t tool_index = 0;
    size_t total_text_len = 0;
    bool truncated = stop_reason &&
                     (strcmp(stop_reason, "max_tokens") == 0 ||
                      strcmp(stop_reason, "model_context_window_exceeded") == 0);
    static const char truncated_note[] = "\n\n[Reply truncated: the model reached its output limit.]";
    esp_err_t err = ESP_OK;

    kept = select_replay_blocks(content);
    if (!kept) {
        *out_error_message = dup_printf("Out of memory copying LLM content");
        return ESP_ERR_NO_MEM;
    }

    cJSON_ArrayForEach(block, kept) {
        if (json_type_is(block, "text")) {
            cJSON *text_json = cJSON_GetObjectItem(block, "text");

            if (cJSON_IsString(text_json) && text_json->valuestring) {
                total_text_len += strlen(text_json->valuestring);
            }
        } else if (json_type_is(block, "tool_use")) {
            tool_count++;
        }
    }

    if (truncated && tool_count > 0) {
        /* A tool call cut off by the token limit may carry incomplete input: never run it. */
        cJSON_Delete(kept);
        *out_error_message = dup_printf("Claude's reply hit the output limit (stop_reason=%s) "
                                        "while calling a tool; try a lower effort or a shorter task",
                                        stop_reason);
        return ESP_FAIL;
    }
    if (truncated && total_text_len == 0) {
        cJSON_Delete(kept);
        *out_error_message = dup_printf("Claude reached the output limit (stop_reason=%s) before "
                                        "answering; thinking counts toward the limit, so try a lower effort",
                                        stop_reason);
        return ESP_FAIL;
    }

    if (total_text_len > 0) {
        size_t offset = 0;
        size_t alloc_len = total_text_len + (truncated ? sizeof(truncated_note) : 0) + 1;
        char *text = calloc(1, alloc_len);

        if (!text) {
            cJSON_Delete(kept);
            *out_error_message = dup_printf("Out of memory copying LLM text");
            return ESP_ERR_NO_MEM;
        }

        cJSON_ArrayForEach(block, kept) {
            cJSON *text_json = NULL;
            size_t len = 0;

            if (!json_type_is(block, "text")) {
                continue;
            }
            text_json = cJSON_GetObjectItem(block, "text");
            if (!cJSON_IsString(text_json) || !text_json->valuestring) {
                continue;
            }
            len = strlen(text_json->valuestring);
            memcpy(text + offset, text_json->valuestring, len);
            offset += len;
        }
        if (truncated) {
            memcpy(text + offset, truncated_note, sizeof(truncated_note));
            ESP_LOGW(TAG, "Reply truncated (stop_reason=%s)", stop_reason);
        }
        out_response->text = text;
    }

    if (tool_count > 0) {
        out_response->tool_calls = calloc(tool_count, sizeof(claw_llm_tool_call_t));
        if (!out_response->tool_calls) {
            cJSON_Delete(kept);
            *out_error_message = dup_printf("Out of memory copying tool calls");
            return ESP_ERR_NO_MEM;
        }
        out_response->tool_call_count = tool_count;

        cJSON_ArrayForEach(block, kept) {
            cJSON *dst_input = NULL;
            claw_llm_tool_call_t *dst = NULL;

            if (!json_type_is(block, "tool_use")) {
                continue;
            }

            dst = &out_response->tool_calls[tool_index];
            err = dup_json_string(cJSON_GetObjectItem(block, "id"), &dst->id);
            if (err == ESP_OK) {
                err = dup_json_string(cJSON_GetObjectItem(block, "name"), &dst->name);
            }
            if (err == ESP_OK) {
                dst_input = cJSON_GetObjectItem(block, "input");
                dst->arguments_json = dst_input ? cJSON_PrintUnformatted(dst_input) : strdup("{}");
                if (!dst->arguments_json) {
                    err = ESP_ERR_NO_MEM;
                }
            }
            if (err != ESP_OK) {
                cJSON_Delete(kept);
                *out_error_message = dup_printf("Malformed or oversized tool call in LLM response");
                return err;
            }
            tool_index++;
        }

        /*
         * The tool loop continues with another request: keep the raw turn,
         * thinking blocks and signatures included, for verbatim replay.
         */
        if (prefix_hex) {
            cJSON *provider_content = cJSON_CreateObject();

            if (!provider_content) {
                cJSON_Delete(kept);
                *out_error_message = dup_printf("Out of memory keeping LLM content");
                return ESP_ERR_NO_MEM;
            }
            cJSON_AddStringToObject(provider_content, "format", CLAW_LLM_ANTHROPIC_CONTENT_FORMAT);
            cJSON_AddStringToObject(provider_content, "prefix", prefix_hex);
            cJSON_AddNumberToObject(provider_content, "seq", (double)ctx->next_seq++);
            cJSON_AddItemToObject(provider_content, "content", kept);
            kept = NULL;
            out_response->provider_content = provider_content;
        }
    }

    cJSON_Delete(kept);
    if (!out_response->text && out_response->tool_call_count == 0) {
        *out_error_message = dup_printf("LLM returned empty response (stop_reason=%s)",
                                        stop_reason ? stop_reason : "unknown");
        return ESP_FAIL;
    }
    return ESP_OK;
}

static esp_err_t anthropic_post(const anthropic_backend_ctx_t *ctx,
                                const claw_llm_model_profile_t *profile,
                                const char *post_data,
                                claw_llm_http_response_t *out_http_response,
                                char **out_error_message)
{
    claw_llm_http_json_request_t http_request = {0};
    char *url = NULL;
    esp_err_t err;
    const claw_llm_http_header_t headers[] = {
        { .name = "x-api-key", .value = ctx->api_key },
        { .name = "anthropic-version", .value = CLAW_LLM_ANTHROPIC_VERSION },
        { .name = "anthropic-beta", .value = CLAW_LLM_ANTHROPIC_FALLBACK_BETA },
    };

    url = join_url(ctx->base_url, profile->chat_path);
    if (!url) {
        *out_error_message = dup_printf("Out of memory building API URL");
        return ESP_ERR_NO_MEM;
    }

    http_request.url = url;
    http_request.body = post_data;
    http_request.auth_type = "none";
    http_request.timeout_ms = ctx->timeout_ms;
    http_request.headers = headers;
    /* The beta header is sent only together with the fallbacks field. */
    http_request.header_count = ctx->server_fallback ? 3 : 2;

    err = claw_llm_http_post_json(&http_request, out_http_response, out_error_message);
    free(url);
    return err;
}

/*
 * Parses one Messages API response. On pause_turn the content is appended to
 * *io_paused_content and ESP_ERR_NOT_FINISHED is returned so the caller can
 * resume the turn.
 */
static esp_err_t handle_response_body(anthropic_backend_ctx_t *ctx,
                                      const char *body,
                                      const char *prefix_hex,
                                      cJSON **io_paused_content,
                                      claw_llm_response_t *out_response,
                                      char **out_error_message)
{
    cJSON *root = NULL;
    cJSON *content = NULL;
    cJSON *stop_json = NULL;
    const char *stop_reason = NULL;
    esp_err_t err;

    root = cJSON_Parse(body);
    if (!root) {
        *out_error_message = dup_printf("Failed to parse LLM JSON response");
        return ESP_FAIL;
    }

    stop_json = cJSON_GetObjectItemCaseSensitive(root, "stop_reason");
    stop_reason = cJSON_IsString(stop_json) ? stop_json->valuestring : NULL;
    log_usage(ctx, root, stop_reason);

    /* Check refusal before reading content: content is empty or partial. */
    if (stop_reason && strcmp(stop_reason, "refusal") == 0) {
        *out_error_message = build_refusal_message(root);
        ESP_LOGW(TAG, "%s", *out_error_message ? *out_error_message : "refusal");
        cJSON_Delete(root);
        return ESP_FAIL;
    }

    content = cJSON_GetObjectItemCaseSensitive(root, "content");
    if (!cJSON_IsArray(content)) {
        cJSON_Delete(root);
        *out_error_message = dup_printf("LLM response missing content");
        return ESP_FAIL;
    }

    if (*io_paused_content) {
        cJSON *block = NULL;

        cJSON_ArrayForEach(block, content) {
            cJSON *dup = cJSON_Duplicate(block, true);

            if (!dup) {
                cJSON_Delete(root);
                *out_error_message = dup_printf("Out of memory resuming paused turn");
                return ESP_ERR_NO_MEM;
            }
            cJSON_AddItemToArray(*io_paused_content, dup);
        }
        content = *io_paused_content;
    }

    if (stop_reason && strcmp(stop_reason, "pause_turn") == 0) {
        if (!*io_paused_content) {
            *io_paused_content = cJSON_DetachItemFromObjectCaseSensitive(root, "content");
        }
        cJSON_Delete(root);
        return ESP_ERR_NOT_FINISHED;
    }

    err = fill_response_from_content(ctx, content, stop_reason, prefix_hex, out_response, out_error_message);
    cJSON_Delete(root);
    return err;
}

static bool is_thinking_signature_error(int status_code, const char *error_message)
{
    return status_code == 400 && error_message &&
           strstr(error_message, "signature") && strstr(error_message, "thinking");
}

/* ---------- Chat ---------- */

static esp_err_t build_chat_body(const anthropic_backend_ctx_t *ctx,
                                 const claw_llm_chat_request_t *request,
                                 int last_user_index,
                                 const char *prefix_hex,
                                 bool strip_all_thinking,
                                 cJSON *paused_content,
                                 bool *out_replays_thinking,
                                 char **out_post_data,
                                 char **out_error_message)
{
    cJSON *body = NULL;
    cJSON *messages = NULL;
    cJSON *tools = NULL;
    char *post_data = NULL;

    body = cJSON_CreateObject();
    if (!body) {
        *out_error_message = dup_printf("Out of memory building request");
        return ESP_ERR_NO_MEM;
    }

    cJSON_AddStringToObject(body, "model", ctx->model);
    cJSON_AddNumberToObject(body, "max_tokens", CLAW_LLM_ANTHROPIC_MAX_TOKENS);

    /*
     * The system prompt is a single cached block. It only changes when its
     * stable inputs change (memory, skills); per-request content travels in
     * the turn context instead.
     */
    if (request->system_prompt && request->system_prompt[0]) {
        cJSON *system = cJSON_AddArrayToObject(body, "system");
        cJSON *system_block = anthropic_make_text_block(request->system_prompt);
        cJSON *cache_control = anthropic_make_cache_control();

        if (!system || !system_block || !cache_control) {
            cJSON_Delete(system_block);
            cJSON_Delete(cache_control);
            cJSON_Delete(body);
            *out_error_message = dup_printf("Out of memory building system prompt");
            return ESP_ERR_NO_MEM;
        }
        cJSON_AddItemToObject(system_block, "cache_control", cache_control);
        cJSON_AddItemToArray(system, system_block);
    }

    messages = convert_messages_to_anthropic(ctx,
                                             request->messages,
                                             request->turn_context,
                                             last_user_index,
                                             prefix_hex,
                                             strip_all_thinking,
                                             out_replays_thinking);
    if (!messages) {
        cJSON_Delete(body);
        *out_error_message = dup_printf("Out of memory converting messages");
        return ESP_ERR_NO_MEM;
    }
    if (paused_content) {
        /* Resume a pause_turn: send the paused assistant turn back as-is. */
        cJSON *paused_message = cJSON_CreateObject();
        cJSON *paused_blocks = cJSON_Duplicate(paused_content, true);

        if (!paused_message || !paused_blocks) {
            cJSON_Delete(paused_message);
            cJSON_Delete(paused_blocks);
            cJSON_Delete(messages);
            cJSON_Delete(body);
            *out_error_message = dup_printf("Out of memory resuming paused turn");
            return ESP_ERR_NO_MEM;
        }
        cJSON_AddStringToObject(paused_message, "role", "assistant");
        cJSON_AddItemToObject(paused_message, "content", paused_blocks);
        cJSON_AddItemToArray(messages, paused_message);
    } else if (!ctx->official_api) {
        /*
         * Gateways may not support top-level automatic caching: mark the last
         * block of the conversation explicitly instead. Moving a cache_control
         * marker does not edit the conversation for preserved thinking.
         */
        cJSON *last_message = cJSON_GetArrayItem(messages, cJSON_GetArraySize(messages) - 1);
        cJSON *last_blocks = cJSON_GetObjectItem(last_message, "content");
        cJSON *last_block = cJSON_GetArrayItem(last_blocks, cJSON_GetArraySize(last_blocks) - 1);

        if (last_block && !block_is_thinking(last_block)) {
            cJSON *cache_control = anthropic_make_cache_control();

            if (cache_control) {
                cJSON_AddItemToObject(last_block, "cache_control", cache_control);
            }
        }
    }
    cJSON_AddItemToObject(body, "messages", messages);

    tools = convert_tools_to_anthropic(request->tools_json);
    if (request->tools_json && request->tools_json[0] && !tools) {
        cJSON_Delete(body);
        *out_error_message = dup_printf("Invalid tools JSON");
        return ESP_ERR_INVALID_ARG;
    }
    if (tools && cJSON_GetArraySize(tools) > 0) {
        cJSON *tool_choice = cJSON_CreateObject();

        if (!tool_choice) {
            cJSON_Delete(tools);
            cJSON_Delete(body);
            *out_error_message = dup_printf("Out of memory building tool choice");
            return ESP_ERR_NO_MEM;
        }
        cJSON_AddItemToObject(body, "tools", tools);
        /* Current models reject forced tool_choice (any/tool); auto is required. */
        cJSON_AddStringToObject(tool_choice, "type", "auto");
        cJSON_AddItemToObject(body, "tool_choice", tool_choice);
    } else {
        cJSON_Delete(tools);
    }

    add_common_request_fields(ctx, body);
    if (ctx->official_api) {
        /* Automatic caching: the breakpoint follows the growing conversation tail. */
        cJSON *cache_control = anthropic_make_cache_control();

        if (cache_control) {
            cJSON_AddItemToObject(body, "cache_control", cache_control);
        }
    }

    post_data = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);
    if (!post_data) {
        *out_error_message = dup_printf("Out of memory serializing request");
        return ESP_ERR_NO_MEM;
    }

    *out_post_data = post_data;
    return ESP_OK;
}

static esp_err_t anthropic_init(const claw_llm_runtime_config_t *config,
                                const claw_llm_model_profile_t *profile,
                                void **out_backend_ctx,
                                char **out_error_message)
{
    anthropic_backend_ctx_t *ctx = NULL;
    const char *base_url = NULL;
    const char *effort = NULL;

    if (!config || !profile || !out_backend_ctx || !out_error_message) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!config->api_key || !config->api_key[0]) {
        *out_error_message = dup_printf("LLM API key is empty");
        return ESP_ERR_INVALID_ARG;
    }
    if (!config->model || !config->model[0]) {
        *out_error_message = dup_printf("LLM model is empty");
        return ESP_ERR_INVALID_ARG;
    }

    base_url = (config->base_url && config->base_url[0]) ? config->base_url : profile->default_base_url;
    if (!base_url || !base_url[0]) {
        *out_error_message = dup_printf("LLM base_url is empty");
        return ESP_ERR_INVALID_ARG;
    }

    if (config->effort && config->effort[0]) {
        if (!is_valid_effort(config->effort)) {
            ESP_LOGW(TAG, "Ignoring unknown effort '%s' (use low, medium, high, xhigh or max)",
                     config->effort);
        } else if (string_in_list(config->model, s_no_effort_model_prefixes,
                                  sizeof(s_no_effort_model_prefixes) / sizeof(s_no_effort_model_prefixes[0]),
                                  true)) {
            ESP_LOGW(TAG, "Model %s does not support effort; ignoring effort '%s'",
                     config->model, config->effort);
        } else {
            effort = config->effort;
        }
    }

    ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        *out_error_message = dup_printf("Out of memory allocating backend context");
        return ESP_ERR_NO_MEM;
    }

    ctx->api_key = strdup(config->api_key);
    ctx->model = strdup(config->model);
    ctx->base_url = strdup(base_url);
    ctx->effort = effort ? strdup(effort) : NULL;
    ctx->timeout_ms = config->timeout_ms ? config->timeout_ms : profile->default_timeout_ms;
    ctx->image_max_bytes = config->image_max_bytes ? config->image_max_bytes : profile->default_image_max_bytes;
    ctx->official_api = is_official_api_url(base_url);
    ctx->server_fallback = ctx->official_api &&
                           string_in_list(config->model, s_server_fallback_models,
                                          sizeof(s_server_fallback_models) / sizeof(s_server_fallback_models[0]),
                                          false);
    if (!ctx->api_key || !ctx->model || !ctx->base_url || (effort && !ctx->effort)) {
        free(ctx->api_key);
        free(ctx->model);
        free(ctx->base_url);
        free(ctx->effort);
        free(ctx);
        *out_error_message = dup_printf("Out of memory copying backend config");
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "model=%s effort=%s caching=%s refusal_fallback=%s",
             ctx->model,
             ctx->effort ? ctx->effort : "(model default)",
             ctx->official_api ? "automatic+system" : "explicit",
             ctx->server_fallback ? "default" : "off");
    *out_backend_ctx = ctx;
    return ESP_OK;
}

static esp_err_t anthropic_chat(void *backend_ctx,
                                const claw_llm_model_profile_t *profile,
                                const claw_llm_chat_request_t *request,
                                claw_llm_response_t *out_response,
                                char **out_error_message)
{
    anthropic_backend_ctx_t *ctx = (anthropic_backend_ctx_t *)backend_ctx;
    claw_llm_http_response_t http_response = {0};
    char prefix_hex[CLAW_LLM_ANTHROPIC_PREFIX_HEX_LEN + 1] = {0};
    cJSON *paused_content = NULL;
    char *post_data = NULL;
    bool strip_all_thinking = false;
    bool replays_thinking = false;
    int last_user_index;
    int continuations = 0;
    esp_err_t err;

    if (!ctx || !profile || !request || !request->messages || !out_response || !out_error_message) {
        return ESP_ERR_INVALID_ARG;
    }
    memset(out_response, 0, sizeof(*out_response));

    last_user_index = find_last_user_message(request->messages);
    err = compute_prefix_fingerprint(request, last_user_index, prefix_hex);
    if (err != ESP_OK) {
        *out_error_message = dup_printf("Out of memory fingerprinting request");
        return err;
    }

    while (true) {
        err = build_chat_body(ctx, request, last_user_index, prefix_hex, strip_all_thinking,
                              paused_content, &replays_thinking, &post_data, out_error_message);
        if (err != ESP_OK) {
            break;
        }

        err = anthropic_post(ctx, profile, post_data, &http_response, out_error_message);
        free(post_data);
        post_data = NULL;

        if (err != ESP_OK && replays_thinking && !strip_all_thinking &&
                is_thinking_signature_error(http_response.status_code, *out_error_message)) {
            /*
             * The API rejected a replayed thinking block (the conversation it
             * was bound to changed). Documented recovery: drop all thinking
             * blocks from the history and retry once. Turns produced so far
             * stay stripped for the rest of the session.
             */
            ESP_LOGW(TAG, "Replayed thinking rejected (%s); retrying without earlier thinking",
                     *out_error_message);
            free(*out_error_message);
            *out_error_message = NULL;
            claw_llm_http_response_free(&http_response);
            ctx->strip_before_seq = ctx->next_seq;
            strip_all_thinking = true;
            continue;
        }
        if (err != ESP_OK) {
            break;
        }

        err = handle_response_body(ctx, http_response.body, prefix_hex, &paused_content,
                                   out_response, out_error_message);
        claw_llm_http_response_free(&http_response);
        if (err == ESP_ERR_NOT_FINISHED) {
            if (++continuations > CLAW_LLM_ANTHROPIC_MAX_PAUSE_CONTINUATIONS) {
                *out_error_message = dup_printf("LLM turn paused too many times (pause_turn)");
                err = ESP_FAIL;
                break;
            }
            continue;
        }
        break;
    }

    cJSON_Delete(paused_content);
    claw_llm_http_response_free(&http_response);
    if (err != ESP_OK) {
        claw_llm_response_free(out_response);
    }
    return err;
}

static esp_err_t anthropic_infer_media(void *backend_ctx,
                                       const claw_llm_model_profile_t *profile,
                                       const claw_llm_media_request_t *request,
                                       char **out_text,
                                       char **out_error_message)
{
    anthropic_backend_ctx_t *ctx = (anthropic_backend_ctx_t *)backend_ctx;
    claw_media_prepared_t prepared = {0};
    claw_llm_response_t response = {0};
    claw_llm_http_response_t http_response = {0};
    cJSON *body = NULL;
    cJSON *messages = NULL;
    cJSON *user_msg = NULL;
    cJSON *content = NULL;
    cJSON *text_block = NULL;
    cJSON *image_block = NULL;
    cJSON *source = NULL;
    cJSON *paused_content = NULL;
    char *mime = NULL;
    char *base64_data = NULL;
    char *post_data = NULL;
    esp_err_t err;

    if (out_text) {
        *out_text = NULL;
    }
    if (out_error_message) {
        *out_error_message = NULL;
    }
    if (!ctx || !profile || !request || !out_text || !out_error_message) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!profile->supports_vision) {
        *out_error_message = dup_printf("Selected profile does not support media inference");
        return ESP_ERR_NOT_SUPPORTED;
    }
    if (!request->user_prompt || !request->user_prompt[0] || !request->media || request->media_count == 0) {
        *out_error_message = dup_printf("media request is incomplete");
        return ESP_ERR_INVALID_ARG;
    }

    err = claw_media_prepare_asset(&request->media[0],
                                   profile,
                                   ctx->image_max_bytes,
                                   &prepared,
                                   out_error_message);
    if (err != ESP_OK) {
        return err;
    }
    if (prepared.kind != CLAW_MEDIA_PREPARED_KIND_DATA_URL) {
        err = ESP_ERR_NOT_SUPPORTED;
        *out_error_message = dup_printf("Anthropic backend requires local image data");
        goto cleanup;
    }

    err = parse_data_url(prepared.payload, &mime, &base64_data);
    if (err != ESP_OK) {
        *out_error_message = dup_printf("Failed to prepare Anthropic image payload");
        goto cleanup;
    }

    body = cJSON_CreateObject();
    messages = cJSON_CreateArray();
    user_msg = cJSON_CreateObject();
    content = cJSON_CreateArray();
    text_block = cJSON_CreateObject();
    image_block = cJSON_CreateObject();
    source = cJSON_CreateObject();
    if (!body || !messages || !user_msg || !content || !text_block || !image_block || !source) {
        err = ESP_ERR_NO_MEM;
        *out_error_message = dup_printf("Out of memory building media request");
        goto cleanup;
    }

    cJSON_AddStringToObject(body, "model", ctx->model);
    cJSON_AddNumberToObject(body, "max_tokens", CLAW_LLM_ANTHROPIC_MAX_TOKENS);
    if (request->system_prompt && request->system_prompt[0]) {
        cJSON_AddStringToObject(body, "system", request->system_prompt);
    }

    cJSON_AddStringToObject(user_msg, "role", "user");
    cJSON_AddStringToObject(image_block, "type", "image");
    cJSON_AddStringToObject(source, "type", "base64");
    cJSON_AddStringToObject(source, "media_type", mime);
    cJSON_AddStringToObject(source, "data", base64_data);
    cJSON_AddItemToObject(image_block, "source", source);
    source = NULL;
    cJSON_AddItemToArray(content, image_block);
    image_block = NULL;

    cJSON_AddStringToObject(text_block, "type", "text");
    cJSON_AddStringToObject(text_block, "text", request->user_prompt);
    cJSON_AddItemToArray(content, text_block);
    text_block = NULL;

    cJSON_AddItemToObject(user_msg, "content", content);
    content = NULL;
    cJSON_AddItemToArray(messages, user_msg);
    user_msg = NULL;
    cJSON_AddItemToObject(body, "messages", messages);
    messages = NULL;
    add_common_request_fields(ctx, body);

    post_data = cJSON_PrintUnformatted(body);
    if (!post_data) {
        err = ESP_ERR_NO_MEM;
        *out_error_message = dup_printf("Out of memory serializing media request");
        goto cleanup;
    }

    err = anthropic_post(ctx, profile, post_data, &http_response, out_error_message);
    if (err != ESP_OK) {
        goto cleanup;
    }

    err = handle_response_body(ctx, http_response.body, NULL, &paused_content, &response, out_error_message);
    if (err == ESP_ERR_NOT_FINISHED) {
        /* No server tools are declared here, so a paused turn is unexpected. */
        err = ESP_FAIL;
        *out_error_message = dup_printf("LLM media turn paused unexpectedly (pause_turn)");
        goto cleanup;
    }
    if (err != ESP_OK) {
        goto cleanup;
    }
    if (!response.text || !response.text[0]) {
        err = ESP_FAIL;
        *out_error_message = dup_printf("LLM returned empty media response");
        goto cleanup;
    }

    *out_text = response.text;
    response.text = NULL;
    err = ESP_OK;

cleanup:
    free(mime);
    free(base64_data);
    free(post_data);
    cJSON_Delete(body);
    cJSON_Delete(messages);
    cJSON_Delete(user_msg);
    cJSON_Delete(content);
    cJSON_Delete(text_block);
    cJSON_Delete(image_block);
    cJSON_Delete(source);
    cJSON_Delete(paused_content);
    claw_llm_http_response_free(&http_response);
    claw_llm_response_free(&response);
    claw_media_prepared_free(&prepared);
    return err;
}

static void anthropic_deinit(void *backend_ctx)
{
    anthropic_backend_ctx_t *ctx = (anthropic_backend_ctx_t *)backend_ctx;

    if (!ctx) {
        return;
    }

    free(ctx->api_key);
    free(ctx->model);
    free(ctx->base_url);
    free(ctx->effort);
    free(ctx);
}

const claw_llm_backend_vtable_t *claw_llm_backend_anthropic_vtable(void)
{
    static const claw_llm_backend_vtable_t vtable = {
        .id = "anthropic",
        .init = anthropic_init,
        .chat = anthropic_chat,
        .infer_media = anthropic_infer_media,
        .deinit = anthropic_deinit,
        .supports_turn_context = true,
    };

    return &vtable;
}
