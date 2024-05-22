TARGET=$(shell basename "`pwd`")
IPAMIRSOLVER ?= solver2022
DEPS = ../ipamir/maxsat/$(IPAMIRSOLVER)/libipamir$(IPAMIRSOLVER).a src/*rs

all: $(TARGET)

$(TARGET): $(DEPS)
	cargo build --release && cp target/release/$(TARGET) .
