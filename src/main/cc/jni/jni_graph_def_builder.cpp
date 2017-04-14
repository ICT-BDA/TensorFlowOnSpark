#include "include/bda_tensorflow_jni_GraphDefBuilder.h"
#include "util/util.h"


/*
* Class:     tensorflow_jni_GraphDefBuilder
* Method:    allocate
* Signature: ()J
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_GraphDefBuilder_allocate
(JNIEnv * env, jobject obj) {
	GraphDefBuilder * gdb = new GraphDefBuilder;
	SetNativeAddress(env, obj, gdb);
}

/*
* Class:     tensorflow_jni_GraphDefBuilder
* Method:    toGraphDef
* Signature: (Ltensorflow/jni/GraphDef;Ltensorflow/jni/Status;)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_GraphDefBuilder_toGraphDef
(JNIEnv * env, jobject obj, jobject jgd, jobject jstatus) {
	GraphDefBuilder * gdb = (GraphDefBuilder*)GetNativeAddress(env, obj);
	GraphDef * gd = (GraphDef*)GetNativeAddress(env, jgd);
	Status * st = (Status*)GetNativeAddress(env, jstatus);
	*st = gdb->ToGraphDef(gd);
}

/*
* Class:     tensorflow_jni_GraphDefBuilder
* Method:    deallocateMemory
* Signature: (J)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_GraphDefBuilder_deallocateMemory
(JNIEnv *env, jobject obj, jlong address) {
	delete (GraphDefBuilder*)address;
}
