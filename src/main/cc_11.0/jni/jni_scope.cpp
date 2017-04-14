#include "util/util.h"
#include "include/bda_tensorflow_jni_11_Scope.h"

/*
 * Class:     bda_tensorflow_jni_11_Scope
 * Method:    deallocateMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_111_Scope_deallocateMemory
(JNIEnv *env, jobject object, jlong address){
    delete (Scope*)address;
}

/*
 * Class:     bda_tensorflow_jni_11_Scope
 * Method:    allocate
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_111_Scope_allocate
(JNIEnv *env, jobject object){
    Scope* root = new Scope(Scope::NewRootScope());
    SetNativeAddress(env, object, root);
}

/*
 * Class:     bda_tensorflow_jni_11_Scope
 * Method:    toGraphDef
 * Signature: (Lbda/tensorflow/jni/GraphDef;Lbda/tensorflow/jni/Status;)V
 */
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_111_Scope_toGraphDef
(JNIEnv *env, jobject jscope, jobject jdef, jobject js){
    Scope *scope = (Scope*)GetNativeAddress(env, jscope);
	GraphDef * gd = (GraphDef*)GetNativeAddress(env, jdef);
	Status * st = (Status*)GetNativeAddress(env, js);
	*st = scope->ToGraphDef(gd);
	SetNativeAddress(env, jdef, gd);
}

JNIEXPORT void JNICALL Java_bda_tensorflow_jni_111_Scope_toGraph
(JNIEnv *env, jobject jscope, jobject jgraph){
    Scope *scope = (Scope*)GetNativeAddress(env, jscope);
    Graph *g = new Graph(OpRegistry::Global());
    scope->ToGraph(g);
    SetNativeAddress(env, jgraph, g);
}