#include "include/bda_tensorflow_jni_GraphDef.h"
#include "util/util.h"


/*
* Class:     tensorflow_jni_GraphDef
* Method:    allocate
* Signature: ()J
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_GraphDef_allocate
(JNIEnv * env, jobject obj) {
	GraphDef* gd = new GraphDef;
	SetNativeAddress(env, obj, gd);
}

/*
* Class:     tensorflow_jni_GraphDef
* Method:    deallocateMemory
* Signature: (J)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_GraphDef_deallocateMemory
(JNIEnv * env, jobject obj, jlong address) {
	delete (GraphDef*)address;
}
