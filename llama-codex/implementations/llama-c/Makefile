CC := cc
CFLAGS := -std=c11 -O2 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS := -lm

BUILD_DIR := build
TARGET := $(BUILD_DIR)/llama-c
TEST_TARGETS := $(BUILD_DIR)/test_config $(BUILD_DIR)/test_tokenizer $(BUILD_DIR)/test_sampler

SRC := \
	src/main.c \
	src/config.c \
	src/weights.c \
	src/transformer.c \
	src/tokenizer.c \
	src/sampler.c \
	src/generate.c

OBJ := $(SRC:src/%.c=$(BUILD_DIR)/%.o)
TEST_CONFIG_OBJ := $(BUILD_DIR)/tests_test_config.o $(BUILD_DIR)/config.o
TEST_TOKENIZER_OBJ := $(BUILD_DIR)/tests_test_tokenizer.o $(BUILD_DIR)/tokenizer.o
TEST_SAMPLER_OBJ := $(BUILD_DIR)/tests_test_sampler.o $(BUILD_DIR)/sampler.o

.PHONY: all clean test dirs

all: $(TARGET)

test: $(TEST_TARGETS)
	$(BUILD_DIR)/test_config
	$(BUILD_DIR)/test_tokenizer
	$(BUILD_DIR)/test_sampler

dirs:
	mkdir -p $(BUILD_DIR)

$(TARGET): dirs $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_config: dirs $(TEST_CONFIG_OBJ)
	$(CC) $(TEST_CONFIG_OBJ) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_tokenizer: dirs $(TEST_TOKENIZER_OBJ)
	$(CC) $(TEST_TOKENIZER_OBJ) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_sampler: dirs $(TEST_SAMPLER_OBJ)
	$(CC) $(TEST_SAMPLER_OBJ) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/tests_%.o: tests/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
