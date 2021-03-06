/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class bda_tensorflow_jni_GraphDefBuilder */

#ifndef _Included_bda_tensorflow_jni_GraphDefBuilder
#define _Included_bda_tensorflow_jni_GraphDefBuilder
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     bda_tensorflow_jni_GraphDefBuilder
 * Method:    allocate
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_GraphDefBuilder_allocate
  (JNIEnv *, jobject);

/*
 * Class:     bda_tensorflow_jni_GraphDefBuilder
 * Method:    toGraphDef
 * Signature: (Lbda/tensorflow/jni/GraphDef;Lbda/tensorflow/jni/Status;)V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_GraphDefBuilder_toGraphDef
  (JNIEnv *, jobject, jobject, jobject);

/*
 * Class:     bda_tensorflow_jni_GraphDefBuilder
 * Method:    deallocateMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_GraphDefBuilder_deallocateMemory
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
