// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include "util.h"

using namespace tensorflow;

bool SetNativeAddress(JNIEnv *env, jobject object, void* address) {

	if (object == NULL) {
		std::cout << "object is NULL" << std::endl;
		return false;
	}
	/* Get a reference to JVM object class */
	jclass claz = env->GetObjectClass(object);
	if (claz == NULL) {
		std::cout << "unable to get object's class" << std::endl;
		return false;
	}
	/* Locate init(long) method */
	jmethodID methodId = env->GetMethodID(claz, "init", "(J)V");
	if (methodId == NULL) {
		std::cout << "could not locate init() method" << std::endl;
		return false;
	}

	/* associate native object with JVM object */
	env->CallVoidMethod(object, methodId, (long)address);
	if (env->ExceptionCheck()) {
		std::cout << "CallVoidMethod failed" << std::endl;
		return false;
	}
	return true;
}

void* GetNativeAddress(JNIEnv *env, jobject object) {
	if (object == NULL) {
		std::cout << "object is NULL" << std::endl;
		return 0;
	}
	/* Get a reference to JVM object class */
	jclass claz = env->GetObjectClass(object);
	if (claz == NULL) {
		std::cout << "unable to get object's class" << std::endl;
		return 0;
	}
	/* Getting the field id in the class */
	jfieldID fieldId = env->GetFieldID(claz, "address", "J");
	if (fieldId == NULL) {
		std::cout << "could not locate field 'address'" << std::endl;
		return 0;
	}

	return (void*)env->GetLongField(object, fieldId);
}

void GetStringVector(JNIEnv *env, std::vector<std::string>& vec, jobjectArray array) {
	jsize length = env->GetArrayLength(array);
	for (int i = 0; i < length; i++) {
		jstring addr = (jstring)env->GetObjectArrayElement(array, i);
		const char *cStr = env->GetStringUTFChars(addr, NULL);
		vec.push_back(cStr);
		env->ReleaseStringUTFChars(addr, cStr);
		//Too many local refs could get created due to the loop, so delete them
		//CHECKME:cStr is also a local ref called in loop, but it's not clear if deleting it via DeleteLocalRef deletes the memory pointed by it too
		env->DeleteLocalRef(addr);
	}
}

jobjectArray GetJStringFromVector(JNIEnv *env, std::vector<std::string>& vec) {
	int length = vec.size();
	jclass string_class = env->FindClass("java/lang/String");
	jobjectArray ret = env->NewObjectArray(length, string_class, NULL);
	for (int i = 0; i < length; i++) {
		std::string str = vec[i];
		jstring js = env->NewStringUTF(str.data());
		env->SetObjectArrayElement(ret, i, js);
		env->DeleteLocalRef(js);
	}
	return ret;
}

void println(std::string line) {
	std::cout << line << std::endl;
}

void putJTensorArrayIntoVector(JNIEnv *env, jobjectArray array, std::vector<Tensor>& vec) {
	vec.clear();
	jsize length = env->GetArrayLength(array);
	for (int i = 0; i < length; i++) {
		jobject element = env->GetObjectArrayElement(array, i);
		Tensor* t = (Tensor*)GetNativeAddress(env, element);
		vec.push_back(*t);		
		//Too many local refs could get created due to the loop, so delete them
		//CHECKME:cStr is also a local ref called in loop, but it's not clear if deleting it via DeleteLocalRef deletes the memory pointed by it too
		env->DeleteLocalRef(element);
	}
}
jobject createNode(JNIEnv *env, Node* node) {
	//std::cout << node << " notice " << node->name()<< std::endl;
	jclass node_class = env->FindClass("bda/tensorflow/jni/Node");
	jmethodID node_constructor = env->GetMethodID(node_class, "<init>", "()V");
	jobject ret = env->NewObject(node_class, node_constructor);
	SetNativeAddress(env, ret, node);
	return ret;
}

void putJintArrIntoVector(JNIEnv *env, jintArray jia, std::vector<int>& vec) {
	vec.clear();
	jsize size = env->GetArrayLength(jia);
	jint* ji_p = env->GetIntArrayElements(jia, 0);
	for (jsize i = 0; i < size; i++) {
		vec.push_back(*(ji_p + i));
	}
	env->ReleaseIntArrayElements(jia, ji_p, 0);
}

void putJfloatArrIntoVector(JNIEnv *env, jfloatArray jfa, std::vector<float>& vec) {
	vec.clear();
	jsize size = env->GetArrayLength(jfa);
	jfloat* jf_p = env->GetFloatArrayElements(jfa, 0);
	for (jsize i = 0; i < size; i++) {
		vec.push_back(*(jf_p + i));
	}
	env->ReleaseFloatArrayElements(jfa, jf_p, 0);
}

void putJNodeoutArrayIntoVector(JNIEnv *env, 
	jobjectArray jnoa,
	std::vector<::tensorflow::NodeOut>& vec) {
	vec.clear();
	jclass nodeout_class = env->FindClass("bda/tensorflow/run/NodeOut");
	jfieldID node_field = env->GetFieldID(nodeout_class, "node", "bda/tensorflow/jni/Node;");
	jfieldID index_field = env->GetFieldID(nodeout_class, "index", "I");
	jsize size = env->GetArrayLength(jnoa);
	for (int i = 0; i < size; i++) {
		jobject jn_o = env->GetObjectArrayElement(jnoa, i);
		jobject jnode = env->GetObjectField(jn_o, node_field);
		Node* node = (Node*)GetNativeAddress(env, jnode);
		jint index = env->GetIntField(jn_o, index_field);
		tensorflow::NodeOut nodeout;
		nodeout.node = node;
		nodeout.index = index;
		vec.push_back(nodeout);
		env->DeleteLocalRef(jn_o);
		env->DeleteLocalRef(jnode);
	}
}

void throwRuntimeException(JNIEnv *env, std::string message){
    jclass exClass;
    string className="bda/tensorflow/exception/TensorflowRuntimeException";

    exClass = env->FindClass(className.c_str());
    return env->ThrowNew(exClass, message.c_str());
}

void throwNodeCreateException(JNIEnv *env, std::string message){
    jclass exClass;
    string className="bda/tensorflow/exception/NodeCreateException";

    exClass = env->FindClass(className.c_str());
    return env->ThrowNew(exClass, message.c_str());
}