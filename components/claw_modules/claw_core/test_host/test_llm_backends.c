/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Host-side test for the LLM request builders / response parsers.
 *
 * Runs the real claw_llm_runtime + backends against recorded Messages API
 * responses through a fake HTTP transport, and checks what goes on the wire:
 * thinking blocks are replayed byte-for-byte within a tool loop, dropped when
 * the prefix they are bound to changed, strip-and-retry on a signature 400,
 * caching / effort / fallback fields, stop_reason handling, and that the
 * OpenAI-compatible backend keeps receiving the merged system prompt.
 *
 * Build and run on a host with a C compiler (Linux, macOS, WSL):
 *     make -C components/claw_modules/claw_core/test_host IDF_PATH=$IDF_PATH run
 */
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "llm/claw_llm_http_transport.h"
#include "llm/claw_llm_runtime.h"
#include "llm/media/claw_media_pipeline.h"

/* ---------- Fake transport ---------- */

#define MAX_EXCHANGES 8

typedef struct {
    int status;
    const char *body;
} fake_reply_t;

static fake_reply_t s_replies[MAX_EXCHANGES];
static size_t s_reply_count;
static size_t s_reply_next;
static char *s_sent_bodies[MAX_EXCHANGES];
static bool s_sent_beta_header[MAX_EXCHANGES];
static size_t s_sent_count;

static void fake_reset(void)
{
    for (size_t i = 0; i < s_sent_count; i++) {
        free(s_sent_bodies[i]);
        s_sent_bodies[i] = NULL;
    }
    s_sent_count = 0;
    s_reply_count = 0;
    s_reply_next = 0;
}

static void fake_queue(int status, const char *body)
{
    s_replies[s_reply_count].status = status;
    s_replies[s_reply_count].body = body;
    s_reply_count++;
}

esp_err_t claw_llm_http_post_json(const claw_llm_http_json_request_t *request,
                                  claw_llm_http_response_t *out_response,
                                  char **out_error_message)
{
    const fake_reply_t *reply;

    memset(out_response, 0, sizeof(*out_response));
    *out_error_message = NULL;
    s_sent_bodies[s_sent_count] = strdup(request->body);
    s_sent_beta_header[s_sent_count] = false;
    for (size_t i = 0; i < request->header_count; i++) {
        if (strcmp(request->headers[i].name, "anthropic-beta") == 0) {
            s_sent_beta_header[s_sent_count] = true;
        }
    }
    s_sent_count++;

    if (s_reply_next >= s_reply_count) {
        *out_error_message = strdup("no recorded reply");
        return ESP_FAIL;
    }
    reply = &s_replies[s_reply_next++];
    out_response->status_code = reply->status;
    if (reply->status != 200) {
        cJSON *root = cJSON_Parse(reply->body);
        cJSON *message = cJSON_GetObjectItem(cJSON_GetObjectItem(root, "error"), "message");
        size_t len = strlen(message->valuestring) + 16;

        *out_error_message = calloc(1, len);
        snprintf(*out_error_message, len, "HTTP %d: %s", reply->status, message->valuestring);
        cJSON_Delete(root);
        return ESP_FAIL;
    }
    out_response->body = strdup(reply->body);
    return ESP_OK;
}

void claw_llm_http_response_free(claw_llm_http_response_t *response)
{
    free(response->body);
    memset(response, 0, sizeof(*response));
}

void claw_llm_http_arm_abort(volatile bool *flag)
{
    (void)flag;
}

void claw_llm_http_disarm_abort(void)
{
}

esp_err_t claw_media_prepare_asset(const claw_media_asset_t *asset,
                                   const claw_llm_model_profile_t *profile,
                                   size_t image_max_bytes,
                                   claw_media_prepared_t *out_prepared,
                                   char **out_error_message)
{
    (void)asset;
    (void)profile;
    (void)image_max_bytes;
    (void)out_prepared;
    *out_error_message = strdup("media not available in host test");
    return ESP_ERR_NOT_SUPPORTED;
}

void claw_media_prepared_free(claw_media_prepared_t *prepared)
{
    (void)prepared;
}

/* ---------- Helpers ---------- */

static int s_failures;

#define CHECK(cond) do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            s_failures++; \
        } \
    } while (0)

static cJSON *sent_json(size_t index)
{
    return index < s_sent_count ? cJSON_Parse(s_sent_bodies[index]) : NULL;
}

static cJSON *path(cJSON *root, const char *fmt, ...)
{
    /* Dotted path; numeric segments index arrays, e.g. "messages.1.content.0". */
    char buf[256];
    char *save = NULL;
    char *seg;
    va_list args;

    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    for (seg = strtok_r(buf, ".", &save); seg && root; seg = strtok_r(NULL, ".", &save)) {
        if (seg[0] >= '0' && seg[0] <= '9') {
            root = cJSON_GetArrayItem(root, atoi(seg));
        } else {
            root = cJSON_GetObjectItemCaseSensitive(root, seg);
        }
    }
    return root;
}

static const char *path_str(cJSON *root, const char *p)
{
    cJSON *item = path(root, "%s", p);

    return cJSON_IsString(item) ? item->valuestring : NULL;
}

static bool str_eq(const char *a, const char *b)
{
    return a && b && strcmp(a, b) == 0;
}

static int count_type(cJSON *blocks, const char *type)
{
    cJSON *block = NULL;
    int count = 0;

    cJSON_ArrayForEach(block, blocks) {
        if (str_eq(path_str(block, "type"), type)) {
            count++;
        }
    }
    return count;
}

static claw_llm_runtime_t *make_runtime(const char *profile, const char *model,
                                        const char *base_url, const char *effort)
{
    claw_llm_runtime_t *runtime = NULL;
    char *error = NULL;
    claw_llm_runtime_config_t config = {
        .api_key = "test-key",
        .profile = profile,
        .model = model,
        .base_url = base_url,
        .timeout_ms = 1000,
        .effort = effort,
    };

    if (claw_llm_runtime_init(&runtime, &config, &error) != ESP_OK) {
        fprintf(stderr, "runtime init failed: %s\n", error ? error : "?");
        exit(2);
    }
    free(error);
    return runtime;
}

/* Mirrors claw_core: appends the assistant tool-call turn and its tool results. */
static void append_tool_turn(cJSON *messages, const claw_llm_response_t *response)
{
    cJSON *assistant = cJSON_CreateObject();
    cJSON *tool_calls = cJSON_AddArrayToObject(assistant, "tool_calls");

    cJSON_AddStringToObject(assistant, "role", "assistant");
    cJSON_AddNullToObject(assistant, "content");
    for (size_t i = 0; i < response->tool_call_count; i++) {
        cJSON *call = cJSON_CreateObject();
        cJSON *function = cJSON_AddObjectToObject(call, "function");

        cJSON_AddStringToObject(call, "id", response->tool_calls[i].id);
        cJSON_AddStringToObject(call, "type", "function");
        cJSON_AddStringToObject(function, "name", response->tool_calls[i].name);
        cJSON_AddStringToObject(function, "arguments", response->tool_calls[i].arguments_json);
        cJSON_AddItemToArray(tool_calls, call);
    }
    if (response->provider_content) {
        cJSON_AddItemToObject(assistant, CLAW_LLM_PROVIDER_CONTENT_KEY,
                              cJSON_Duplicate(response->provider_content, true));
    }
    cJSON_AddItemToArray(messages, assistant);

    for (size_t i = 0; i < response->tool_call_count; i++) {
        cJSON *tool = cJSON_CreateObject();

        cJSON_AddStringToObject(tool, "role", "tool");
        cJSON_AddStringToObject(tool, "tool_call_id", response->tool_calls[i].id);
        cJSON_AddStringToObject(tool, "content", "ok");
        cJSON_AddItemToArray(messages, tool);
    }
}

/* ---------- Recorded responses ---------- */

static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"led_set\",\"description\":\"Set the LED\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"on\":{\"type\":\"boolean\"}}}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"time_get\",\"description\":\"Get time\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";

/* Thinking (empty text, as under the default display) + progress thinking + two parallel tool calls. */
static const char *RESP_THINKING_TOOL_USE =
    "{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\","
    "\"content\":["
    "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"EuYBCkQYAiJA+/sig\\u00e9A==\"},"
    "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"PROGRESS_SIG_2\"},"
    "{\"type\":\"tool_use\",\"id\":\"toolu_01\",\"name\":\"led_set\",\"input\":{\"on\":true,\"level\":0.25}},"
    "{\"type\":\"tool_use\",\"id\":\"toolu_02\",\"name\":\"time_get\",\"input\":{}}],"
    "\"stop_reason\":\"tool_use\",\"stop_details\":null,"
    "\"usage\":{\"input_tokens\":12,\"cache_creation_input_tokens\":900,\"cache_read_input_tokens\":0,\"output_tokens\":80}}";

static const char *RESP_END_TURN =
    "{\"id\":\"msg_02\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\","
    "\"content\":[{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"SIG_FINAL\"},"
    "{\"type\":\"text\",\"text\":\"The LED is on.\"}],"
    "\"stop_reason\":\"end_turn\",\"usage\":{\"input_tokens\":5,\"cache_read_input_tokens\":1000,\"output_tokens\":9}}";

static const char *RESP_REFUSAL =
    "{\"id\":\"msg_03\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\","
    "\"content\":[],\"stop_reason\":\"refusal\","
    "\"stop_details\":{\"type\":\"refusal\",\"category\":\"cyber\",\"explanation\":null},"
    "\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}";

static const char *RESP_MAX_TOKENS_TOOL =
    "{\"id\":\"msg_04\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\","
    "\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_09\",\"name\":\"led_set\",\"input\":{}}],"
    "\"stop_reason\":\"max_tokens\",\"usage\":{\"input_tokens\":5,\"output_tokens\":16000}}";

static const char *RESP_MAX_TOKENS_TEXT =
    "{\"id\":\"msg_05\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\","
    "\"content\":[{\"type\":\"text\",\"text\":\"Partial answer\"}],"
    "\"stop_reason\":\"max_tokens\",\"usage\":{\"input_tokens\":5,\"output_tokens\":16000}}";

/* Server-side fallback after a mid-output decline (non-streaming: the partial is omitted). */
static const char *RESP_FALLBACK =
    "{\"id\":\"msg_06\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5\","
    "\"content\":[{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"DECLINED_SIG\"},"
    "{\"type\":\"fallback\",\"from\":{\"model\":\"claude-opus-5-5\"},\"to\":{\"model\":\"claude-opus-5\"}},"
    "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"FALLBACK_SIG\"},"
    "{\"type\":\"tool_use\",\"id\":\"toolu_05\",\"name\":\"led_set\",\"input\":{\"on\":false}}],"
    "\"stop_reason\":\"tool_use\",\"usage\":{\"input_tokens\":5,\"output_tokens\":20}}";

static const char *RESP_PAUSE =
    "{\"id\":\"msg_07\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\","
    "\"content\":[{\"type\":\"text\",\"text\":\"Working... \"}],"
    "\"stop_reason\":\"pause_turn\",\"usage\":{\"input_tokens\":5,\"output_tokens\":3}}";

static const char *ERR_SIGNATURE =
    "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":"
    "\"messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation.\"}}";

static const char *RESP_OPENAI =
    "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}";

/* ---------- Tests ---------- */

static cJSON *base_messages(void)
{
    cJSON *messages = cJSON_CreateArray();
    cJSON *history_user = cJSON_CreateObject();
    cJSON *history_assistant = cJSON_CreateObject();
    cJSON *user = cJSON_CreateObject();

    cJSON_AddStringToObject(history_user, "role", "user");
    cJSON_AddStringToObject(history_user, "content", "Hello");
    cJSON_AddStringToObject(history_assistant, "role", "assistant");
    cJSON_AddStringToObject(history_assistant, "content", "Hi there");
    cJSON_AddStringToObject(user, "role", "user");
    cJSON_AddStringToObject(user, "content", "Turn on the LED");
    cJSON_AddItemToArray(messages, history_user);
    cJSON_AddItemToArray(messages, history_assistant);
    cJSON_AddItemToArray(messages, user);
    return messages;
}

static void test_tool_loop_round_trip(void)
{
    claw_llm_runtime_t *runtime = make_runtime("anthropic", "claude-opus-5-5", NULL, "high");
    cJSON *messages = base_messages();
    claw_llm_response_t first = {0};
    claw_llm_response_t second = {0};
    claw_llm_chat_request_t request = {
        .system_prompt = "You are ESP-Claw.",
        .messages = messages,
        .tools_json = TOOLS_JSON,
        .turn_context = "## Core Request\n- request_id: 7",
    };
    char *error = NULL;
    cJSON *body;
    cJSON *recorded;
    char *recorded_content;
    char *replayed_content;

    fake_reset();
    fake_queue(200, RESP_THINKING_TOOL_USE);
    fake_queue(200, RESP_END_TURN);

    CHECK(claw_llm_runtime_chat(runtime, &request, &first, &error) == ESP_OK);
    CHECK(first.tool_call_count == 2);
    CHECK(first.text == NULL);
    CHECK(first.provider_content != NULL);
    CHECK(str_eq(first.tool_calls[0].arguments_json, "{\"on\":true,\"level\":0.25}"));

    body = sent_json(0);
    CHECK(str_eq(path_str(body, "model"), "claude-opus-5-5"));
    CHECK(str_eq(path_str(body, "system.0.text"), "You are ESP-Claw."));
    CHECK(str_eq(path_str(body, "system.0.cache_control.type"), "ephemeral"));
    CHECK(str_eq(path_str(body, "cache_control.type"), "ephemeral"));
    CHECK(str_eq(path_str(body, "output_config.effort"), "high"));
    CHECK(str_eq(path_str(body, "fallbacks"), "default"));
    CHECK(s_sent_beta_header[0]);
    CHECK(str_eq(path_str(body, "tool_choice.type"), "auto"));
    CHECK(path(body, "temperature") == NULL);
    CHECK(path(body, "top_p") == NULL);
    CHECK(path(body, "thinking") == NULL);
    CHECK(path(body, "max_tokens")->valueint == 16000);
    CHECK(str_eq(path_str(body, "tools.0.name"), "led_set"));
    /* History stays plain text; turn context leads the current user turn. */
    CHECK(str_eq(path_str(body, "messages.0.content.0.text"), "Hello"));
    CHECK(str_eq(path_str(body, "messages.1.content.0.text"), "Hi there"));
    CHECK(str_eq(path_str(body, "messages.2.content.0.text"), "## Core Request\n- request_id: 7"));
    CHECK(str_eq(path_str(body, "messages.2.content.1.text"), "Turn on the LED"));
    cJSON_Delete(body);

    append_tool_turn(messages, &first);
    CHECK(claw_llm_runtime_chat(runtime, &request, &second, &error) == ESP_OK);
    CHECK(str_eq(second.text, "The LED is on."));
    CHECK(second.provider_content == NULL);

    /* The assistant turn is replayed exactly as received, thinking + signatures included. */
    body = sent_json(1);
    recorded = cJSON_Parse(RESP_THINKING_TOOL_USE);
    recorded_content = cJSON_PrintUnformatted(path(recorded, "content"));
    replayed_content = cJSON_PrintUnformatted(path(body, "messages.3.content"));
    CHECK(str_eq(path_str(body, "messages.3.role"), "assistant"));
    CHECK(str_eq(recorded_content, replayed_content));
    CHECK(str_eq(path_str(body, "messages.3.content.0.signature"), "EuYBCkQYAiJA+/sig\xc3\xa9" "A=="));
    /* Parallel tool results travel in one user message. */
    CHECK(cJSON_GetArraySize(path(body, "messages")) == 5);
    CHECK(count_type(path(body, "messages.4.content"), "tool_result") == 2);
    /* Prefix (system, tools, earlier turns) unchanged between the two calls. */
    {
        cJSON *first_body = sent_json(0);
        char *a = cJSON_PrintUnformatted(path(first_body, "system"));
        char *b = cJSON_PrintUnformatted(path(body, "system"));
        char *c = cJSON_PrintUnformatted(path(first_body, "messages.2"));
        char *d = cJSON_PrintUnformatted(path(body, "messages.2"));

        CHECK(str_eq(a, b));
        CHECK(str_eq(c, d));
        free(a);
        free(b);
        free(c);
        free(d);
        cJSON_Delete(first_body);
    }
    free(recorded_content);
    free(replayed_content);
    cJSON_Delete(recorded);
    cJSON_Delete(body);

    /* A system prompt change (e.g. a skill was activated) invalidates earlier thinking: strip it. */
    {
        claw_llm_response_t third = {0};
        claw_llm_chat_request_t changed = request;

        changed.system_prompt = "You are ESP-Claw.\n\n## Active Skill Docs\n...";
        fake_queue(200, RESP_END_TURN);
        CHECK(claw_llm_runtime_chat(runtime, &changed, &third, &error) == ESP_OK);
        body = sent_json(2);
        CHECK(count_type(path(body, "messages.3.content"), "thinking") == 0);
        CHECK(count_type(path(body, "messages.3.content"), "tool_use") == 2);
        cJSON_Delete(body);
        claw_llm_response_free(&third);
    }

    claw_llm_response_free(&first);
    claw_llm_response_free(&second);
    cJSON_Delete(messages);
    claw_llm_runtime_deinit(runtime);
}

static void test_signature_error_strip_and_retry(void)
{
    claw_llm_runtime_t *runtime = make_runtime("anthropic", "claude-sonnet-5-5", NULL, NULL);
    cJSON *messages = base_messages();
    claw_llm_response_t first = {0};
    claw_llm_response_t second = {0};
    claw_llm_response_t third = {0};
    claw_llm_chat_request_t request = {
        .system_prompt = "sys",
        .messages = messages,
        .tools_json = TOOLS_JSON,
    };
    char *error = NULL;
    cJSON *body;

    fake_reset();
    fake_queue(200, RESP_THINKING_TOOL_USE);
    fake_queue(400, ERR_SIGNATURE);
    fake_queue(200, RESP_THINKING_TOOL_USE);
    fake_queue(200, RESP_END_TURN);

    CHECK(claw_llm_runtime_chat(runtime, &request, &first, &error) == ESP_OK);
    body = sent_json(0);
    CHECK(path(body, "output_config") == NULL);
    cJSON_Delete(body);
    append_tool_turn(messages, &first);

    CHECK(claw_llm_runtime_chat(runtime, &request, &second, &error) == ESP_OK);
    CHECK(s_sent_count == 3);
    body = sent_json(1);
    CHECK(count_type(path(body, "messages.3.content"), "thinking") == 2);
    cJSON_Delete(body);
    body = sent_json(2);
    CHECK(count_type(path(body, "messages.3.content"), "thinking") == 0);
    cJSON_Delete(body);

    /* Turns from before the retry stay stripped; the newer turn keeps its thinking. */
    append_tool_turn(messages, &second);
    CHECK(claw_llm_runtime_chat(runtime, &request, &third, &error) == ESP_OK);
    body = sent_json(3);
    CHECK(count_type(path(body, "messages.3.content"), "thinking") == 0);
    CHECK(count_type(path(body, "messages.5.content"), "thinking") == 2);
    cJSON_Delete(body);

    claw_llm_response_free(&first);
    claw_llm_response_free(&second);
    claw_llm_response_free(&third);
    cJSON_Delete(messages);
    claw_llm_runtime_deinit(runtime);
}

static void test_stop_reasons(void)
{
    claw_llm_runtime_t *runtime = make_runtime("anthropic", "claude-opus-5-5", NULL, NULL);
    cJSON *messages = base_messages();
    claw_llm_chat_request_t request = { .system_prompt = "sys", .messages = messages, .tools_json = TOOLS_JSON };
    claw_llm_response_t response = {0};
    char *error = NULL;
    cJSON *body;

    fake_reset();
    fake_queue(200, RESP_REFUSAL);
    CHECK(claw_llm_runtime_chat(runtime, &request, &response, &error) == ESP_FAIL);
    CHECK(error && strstr(error, "declined") && strstr(error, "cyber"));
    free(error);
    error = NULL;

    fake_queue(200, RESP_MAX_TOKENS_TOOL);
    CHECK(claw_llm_runtime_chat(runtime, &request, &response, &error) == ESP_FAIL);
    CHECK(response.tool_call_count == 0);
    CHECK(error && strstr(error, "max_tokens"));
    free(error);
    error = NULL;

    fake_queue(200, RESP_MAX_TOKENS_TEXT);
    CHECK(claw_llm_runtime_chat(runtime, &request, &response, &error) == ESP_OK);
    CHECK(response.text && strncmp(response.text, "Partial answer", 14) == 0 && strstr(response.text, "truncated"));
    claw_llm_response_free(&response);

    /* Only the fallback model's blocks after the marker are kept and run. */
    fake_queue(200, RESP_FALLBACK);
    CHECK(claw_llm_runtime_chat(runtime, &request, &response, &error) == ESP_OK);
    CHECK(response.tool_call_count == 1 && str_eq(response.tool_calls[0].id, "toolu_05"));
    CHECK(count_type(path(response.provider_content, "content"), "fallback") == 0);
    CHECK(count_type(path(response.provider_content, "content"), "thinking") == 1);
    CHECK(str_eq(path_str(response.provider_content, "content.0.signature"), "FALLBACK_SIG"));
    claw_llm_response_free(&response);

    /* pause_turn: resend the paused assistant turn, then merge the continuation. */
    fake_reset();
    fake_queue(200, RESP_PAUSE);
    fake_queue(200, RESP_END_TURN);
    CHECK(claw_llm_runtime_chat(runtime, &request, &response, &error) == ESP_OK);
    CHECK(str_eq(response.text, "Working... The LED is on."));
    body = sent_json(1);
    CHECK(str_eq(path_str(body, "messages.3.role"), "assistant"));
    CHECK(str_eq(path_str(body, "messages.3.content.0.text"), "Working... "));
    cJSON_Delete(body);
    claw_llm_response_free(&response);

    cJSON_Delete(messages);
    claw_llm_runtime_deinit(runtime);
}

static void test_custom_base_url_and_haiku(void)
{
    claw_llm_runtime_t *runtime = make_runtime("anthropic", "claude-haiku-4-5",
                                               "https://gateway.example.com/anthropic/v1", "low");
    cJSON *messages = base_messages();
    claw_llm_chat_request_t request = { .system_prompt = "sys", .messages = messages };
    claw_llm_response_t response = {0};
    char *error = NULL;
    cJSON *body;

    fake_reset();
    fake_queue(200, RESP_END_TURN);
    CHECK(claw_llm_runtime_chat(runtime, &request, &response, &error) == ESP_OK);
    body = sent_json(0);
    CHECK(path(body, "fallbacks") == NULL);
    CHECK(!s_sent_beta_header[0]);
    CHECK(path(body, "cache_control") == NULL);
    CHECK(path(body, "output_config") == NULL); /* Haiku 4.5 rejects effort. */
    CHECK(str_eq(path_str(body, "system.0.cache_control.type"), "ephemeral"));
    CHECK(str_eq(path_str(body, "messages.2.content.0.cache_control.type"), "ephemeral"));
    cJSON_Delete(body);
    claw_llm_response_free(&response);
    cJSON_Delete(messages);
    claw_llm_runtime_deinit(runtime);
}

static void test_openai_backend_unchanged(void)
{
    claw_llm_runtime_t *runtime = make_runtime("openai", "gpt-5.4", NULL, "high");
    cJSON *messages = base_messages();
    cJSON *assistant = cJSON_CreateObject();
    claw_llm_chat_request_t request = {
        .system_prompt = "sys",
        .messages = messages,
        .turn_context = "## Core Request\n- request_id: 7",
    };
    claw_llm_response_t response = {0};
    char *error = NULL;
    cJSON *body;

    cJSON_AddStringToObject(assistant, "role", "assistant");
    cJSON_AddStringToObject(assistant, "content", "x");
    cJSON_AddItemToObject(assistant, CLAW_LLM_PROVIDER_CONTENT_KEY, cJSON_CreateObject());
    cJSON_AddItemToArray(messages, assistant);

    fake_reset();
    fake_queue(200, RESP_OPENAI);
    CHECK(claw_llm_runtime_chat(runtime, &request, &response, &error) == ESP_OK);
    body = sent_json(0);
    CHECK(str_eq(path_str(body, "messages.0.content"), "sys\n\n## Core Request\n- request_id: 7"));
    CHECK(path(body, "messages.4." CLAW_LLM_PROVIDER_CONTENT_KEY) == NULL);
    CHECK(path(body, "output_config") == NULL);
    cJSON_Delete(body);
    claw_llm_response_free(&response);
    cJSON_Delete(messages);
    claw_llm_runtime_deinit(runtime);
}

int main(void)
{
    test_tool_loop_round_trip();
    test_signature_error_strip_and_retry();
    test_stop_reasons();
    test_custom_base_url_and_haiku();
    test_openai_backend_unchanged();
    fake_reset();

    if (s_failures) {
        fprintf(stderr, "%d check(s) failed\n", s_failures);
        return 1;
    }
    printf("All LLM backend host tests passed\n");
    return 0;
}
