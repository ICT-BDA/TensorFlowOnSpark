#pragma once
#include <jni.h>
#include <vector>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <iostream>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/gradients.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/framework/gradients.h"

using namespace tensorflow;
using namespace tensorflow::ops;

bool SetNativeAddress(JNIEnv *env, jobject object, void* address);

void* GetNativeAddress(JNIEnv *env, jobject object);

void GetStringVector(JNIEnv *env, std::vector<std::string>& vec, jobjectArray array);

jobject createNode(JNIEnv *env, Node* node);

void putJintArrIntoVector(JNIEnv *env, jintArray jia, std::vector<int>& vec);

void putJfloatArrIntoVector(JNIEnv *env, jfloatArray jfa, std::vector<float>& vec);

void putJTensorArrayIntoVector(JNIEnv *env, jobjectArray array, std::vector<Tensor>& vec);

void putJNodeoutArrayIntoVector(JNIEnv *env,
	jobjectArray jnoa,
	std::vector<::tensorflow::NodeOut>& vec);

jobjectArray GetJStringFromVector(JNIEnv *env, std::vector<std::string>& vec);

void println(std::string line);

jint throwRuntimeException(JNIEnv *env, std::string message);
jint throwNodeCreateException(JNIEnv *env, std::string message);

jobject createInput(JNIEnv *env, Input *input);
