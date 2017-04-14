#include "include/bda_tensorflow_jni_TensorShape.h"
#include "util/util.h"


/*
* Class:     tensorflow_jni_TensorShape
* Method:    allocate
* Signature: ([I)J
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_TensorShape_allocate
(JNIEnv * env, jobject obj, jintArray jia) {
	std::vector<int> vi;
	putJintArrIntoVector(env, jia, vi);
	std::vector<int64> vi64;
	for (int i : vi) {
		vi64.push_back((int64)i);
	}
	TensorShape* ts = new TensorShape(gtl::ArraySlice<int64>(vi64));
	SetNativeAddress(env, obj, ts);
}

/*
* Class:     tensorflow_jni_TensorShape
* Method:    deallocateMemory
* Signature: (J)V
*/
JNIEXPORT void JNICALL Java_bda_tensorflow_jni_TensorShape_deallocateMemory
(JNIEnv * env, jobject obj, jlong address) {
	delete (TensorShape*)address;
}
