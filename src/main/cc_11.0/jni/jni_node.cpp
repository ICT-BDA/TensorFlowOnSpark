#include "include/bda_tensorflow_jni_Node.h"
#include "util/util.h"

/*
* Class:     tensorflow_jni_Node
* Method:    name
* Signature: ()Ljava/lang/String;
*/
JNIEXPORT jstring JNICALL Java_bda_tensorflow_jni_Node_name
(JNIEnv * env, jobject obj) {
	//println("in node.cpp line 11");
	Node* node = (Node*)GetNativeAddress(env, obj);
	//println("in node.cpp line 13");
	std::cout << node << std::endl;
	std::string name = node->name();
	//println("in node.cpp line 15");
	return env->NewStringUTF(name.data());
}
