#include "include/bda_tensorflow_jni_Status.h"
#include "util/util.h"


/*
* Class:     tensorflow_jni_Status
* Method:    allocate
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Status_allocate
(JNIEnv * env, jobject obj) {
	Status* st = new Status;
	SetNativeAddress(env, obj, st);
}

/*
* Class:     tensorflow_jni_Status
* Method:    deallocateMemory
* Signature: (J)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Status_deallocateMemory
(JNIEnv * env, jobject obj, jlong address) {
	delete (Status*)address;
}

/*
* Class:     tensorflow_jni_Status
* Method:    ok
* Signature: ()Z
*/
JNIEXPORT jboolean JNICALL Java_bda_tensorflow_jni_Status_ok
(JNIEnv *env, jobject obj) {
	Status* sta = (Status*)GetNativeAddress(env, obj);
	return sta->ok();
}

/*
* Class:     tensorflow_jni_Status
* Method:    errorMessage
* Signature: ()Ljava/lang/String;
*/
JNIEXPORT jstring JNICALL Java_bda_tensorflow_jni_Status_errorMessage
(JNIEnv *env, jobject obj) {
	Status* sta = (Status*)GetNativeAddress(env, obj);
	jstring ret = env->NewStringUTF(sta->error_message().data());
	return ret;
}
