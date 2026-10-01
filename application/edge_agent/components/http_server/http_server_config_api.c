/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "http_server_priv.h"

static void http_server_config_to_json(cJSON *root, const app_config_t *config)
{
    http_server_json_add_string(root, "wifi_ssid", config->wifi_ssid);
    http_server_json_add_string(root, "wifi_password", config->wifi_password);
    http_server_json_add_string(root, "llm_api_key", config->llm_api_key);
    http_server_json_add_string(root, "llm_backend_type", config->llm_backend_type);
    http_server_json_add_string(root, "llm_profile", config->llm_profile);
    http_server_json_add_string(root, "llm_model", config->llm_model);
    http_server_json_add_string(root, "llm_base_url", config->llm_base_url);
    http_server_json_add_string(root, "llm_auth_type", config->llm_auth_type);
    http_server_json_add_string(root, "llm_timeout_ms", config->llm_timeout_ms);
    http_server_json_add_string(root, "llm_effort", config->llm_effort);
    http_server_json_add_string(root, "qq_app_id", config->qq_app_id);
    http_server_json_add_string(root, "qq_app_secret", config->qq_app_secret);
    http_server_json_add_string(root, "feishu_app_id", config->feishu_app_id);
    http_server_json_add_string(root, "feishu_app_secret", config->feishu_app_secret);
    http_server_json_add_string(root, "tg_bot_token", config->tg_bot_token);
    http_server_json_add_string(root, "wechat_token", config->wechat_token);
    http_server_json_add_string(root, "wechat_base_url", config->wechat_base_url);
    http_server_json_add_string(root, "wechat_cdn_base_url", config->wechat_cdn_base_url);
    http_server_json_add_string(root, "wechat_account_id", config->wechat_account_id);
    http_server_json_add_string(root, "search_brave_key", config->search_brave_key);
    http_server_json_add_string(root, "search_tavily_key", config->search_tavily_key);
    http_server_json_add_string(root, "enabled_cap_groups", config->enabled_cap_groups);
    http_server_json_add_string(root, "llm_visible_cap_groups", config->llm_visible_cap_groups);
    http_server_json_add_string(root, "enabled_lua_modules", config->enabled_lua_modules);
    http_server_json_add_string(root, "time_timezone", config->time_timezone);
}

static void http_server_config_from_json(app_config_t *config, cJSON *root)
{
    http_server_json_read_string(root, "wifi_ssid", config->wifi_ssid, sizeof(config->wifi_ssid));
    http_server_json_read_string(root, "wifi_password", config->wifi_password, sizeof(config->wifi_password));
    http_server_json_read_string(root, "llm_api_key", config->llm_api_key, sizeof(config->llm_api_key));
    http_server_json_read_string(root, "llm_backend_type", config->llm_backend_type, sizeof(config->llm_backend_type));
    http_server_json_read_string(root, "llm_profile", config->llm_profile, sizeof(config->llm_profile));
    http_server_json_read_string(root, "llm_model", config->llm_model, sizeof(config->llm_model));
    http_server_json_read_string(root, "llm_base_url", config->llm_base_url, sizeof(config->llm_base_url));
    http_server_json_read_string(root, "llm_auth_type", config->llm_auth_type, sizeof(config->llm_auth_type));
    http_server_json_read_string(root, "llm_timeout_ms", config->llm_timeout_ms, sizeof(config->llm_timeout_ms));
    http_server_json_read_string(root, "llm_effort", config->llm_effort, sizeof(config->llm_effort));
    http_server_json_read_string(root, "qq_app_id", config->qq_app_id, sizeof(config->qq_app_id));
    http_server_json_read_string(root, "qq_app_secret", config->qq_app_secret, sizeof(config->qq_app_secret));
    http_server_json_read_string(root, "feishu_app_id", config->feishu_app_id, sizeof(config->feishu_app_id));
    http_server_json_read_string(root, "feishu_app_secret", config->feishu_app_secret, sizeof(config->feishu_app_secret));
    http_server_json_read_string(root, "tg_bot_token", config->tg_bot_token, sizeof(config->tg_bot_token));
    http_server_json_read_string(root, "wechat_token", config->wechat_token, sizeof(config->wechat_token));
    http_server_json_read_string(root, "wechat_base_url", config->wechat_base_url, sizeof(config->wechat_base_url));
    http_server_json_read_string(root, "wechat_cdn_base_url", config->wechat_cdn_base_url, sizeof(config->wechat_cdn_base_url));
    http_server_json_read_string(root, "wechat_account_id", config->wechat_account_id, sizeof(config->wechat_account_id));
    http_server_json_read_string(root, "search_brave_key", config->search_brave_key, sizeof(config->search_brave_key));
    http_server_json_read_string(root, "search_tavily_key", config->search_tavily_key, sizeof(config->search_tavily_key));
    http_server_json_read_string(root, "enabled_cap_groups", config->enabled_cap_groups, sizeof(config->enabled_cap_groups));
    http_server_json_read_string(root, "llm_visible_cap_groups", config->llm_visible_cap_groups, sizeof(config->llm_visible_cap_groups));
    http_server_json_read_string(root, "enabled_lua_modules", config->enabled_lua_modules, sizeof(config->enabled_lua_modules));
    http_server_json_read_string(root, "time_timezone", config->time_timezone, sizeof(config->time_timezone));
}

static esp_err_t config_get_handler(httpd_req_t *req)
{
    http_server_ctx_t *ctx = http_server_ctx();
    app_config_t config;
    esp_err_t err = ctx->services.load_config(&config);
    if (err != ESP_OK) {
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to load config");
    }

    cJSON *root = cJSON_CreateObject();
    if (!root) {
        httpd_resp_send_500(req);
        return ESP_ERR_NO_MEM;
    }

    http_server_config_to_json(root, &config);
    return http_server_send_json_response(req, root);
}

static esp_err_t config_post_handler(httpd_req_t *req)
{
    http_server_ctx_t *ctx = http_server_ctx();
    app_config_t config;
    esp_err_t err = ctx->services.load_config(&config);
    if (err != ESP_OK) {
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to load config");
    }

    cJSON *root = NULL;
    err = http_server_parse_json_body(req, &root);
    if (err != ESP_OK) {
        return httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON body");
    }

    http_server_config_from_json(&config, root);
    cJSON_Delete(root);

    err = ctx->services.save_config(&config);
    if (err != ESP_OK) {
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to save config");
    }

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, max-age=0");
    return httpd_resp_sendstr(req, "{\"ok\":true,\"message\":\"Saved. Restart the device to apply Wi-Fi, core LLM, capability, and Lua module changes.\"}");
}

esp_err_t http_server_register_config_routes(httpd_handle_t server)
{
    const httpd_uri_t handlers[] = {
        { .uri = "/api/config", .method = HTTP_GET, .handler = config_get_handler },
        { .uri = "/api/config", .method = HTTP_POST, .handler = config_post_handler },
    };

    for (size_t i = 0; i < sizeof(handlers) / sizeof(handlers[0]); ++i) {
        esp_err_t err = httpd_register_uri_handler(server, &handlers[i]);
        if (err != ESP_OK) {
            return err;
        }
    }
    return ESP_OK;
}
