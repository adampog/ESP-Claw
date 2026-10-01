# Host-side tests for the claw_core LLM backends (request builders and response parsers).
#
#   make -C components/claw_modules/claw_core/test_host IDF_PATH=$IDF_PATH run
#
# Needs a host C compiler and ESP-IDF's bundled cJSON sources.

IDF_PATH ?= $(HOME)/esp/esp-idf
CJSON_DIR := $(IDF_PATH)/components/json/cJSON
SRC_DIR := ../src
BUILD_DIR := build

CC ?= cc
CFLAGS ?= -std=gnu11 -g -O1 -Wall -fsanitize=address,undefined -fno-omit-frame-pointer
CPPFLAGS += -Istubs -I$(SRC_DIR) -I../include -I$(CJSON_DIR)

SRCS := test_llm_backends.c \
        $(SRC_DIR)/llm/claw_llm_runtime.c \
        $(SRC_DIR)/llm/backends/claw_llm_backend_anthropic.c \
        $(SRC_DIR)/llm/backends/claw_llm_backend_openai_compatible.c \
        $(SRC_DIR)/llm/backends/claw_llm_backend_custom.c \
        $(CJSON_DIR)/cJSON.c

$(BUILD_DIR)/test_llm_backends: $(SRCS) $(wildcard stubs/*.h) $(wildcard $(SRC_DIR)/llm/*.h)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SRCS) -o $@

.PHONY: run clean
run: $(BUILD_DIR)/test_llm_backends
	./$(BUILD_DIR)/test_llm_backends

clean:
	rm -rf $(BUILD_DIR)
