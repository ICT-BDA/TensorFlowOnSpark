/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class bda_tensorflow_jni_Status */

#ifndef _Included_bda_tensorflow_jni_Status
#define _Included_bda_tensorflow_jni_Status
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     bda_tensorflow_jni_Status
 * Method:    allocate
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Status_allocate
  (JNIEnv *, jobject);

/*
 * Class:     bda_tensorflow_jni_Status
 * Method:    ok
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_bda_tensorflow_jni_Status_ok
  (JNIEnv *, jobject);

/*
 * Class:     bda_tensorflow_jni_Status
 * Method:    errorMessage
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_bda_tensorflow_jni_Status_errorMessage
  (JNIEnv *, jobject);

/*
 * Class:     bda_tensorflow_jni_Status
 * Method:    deallocateMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_Status_deallocateMemory
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
