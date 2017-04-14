#include "include/bda_tensorflow_jni_input_Input.h"
#include "util/util.h"


static io::InputBuffer* createInputBuffer(const std::string file_name) {
	Env* env = Env::Default();
	RandomAccessFile* file = nullptr;
	env->NewRandomAccessFile(file_name, &file);
	return new io::InputBuffer(file, 256 << 10);
}


/*
* Class:     tensorflow_jni_input_Input
* Method:    createTensorArrayFromFile
* Signature: (Ljava/lang/String;)[Ltensorflow/jni/input/Record;
*/
JNIEXPORT jobjectArray JNICALL Java_bda_tensorflow_jni_input_Input_createImageAndLabelArrayFromFile
(JNIEnv * env, jclass cls, jstring file_path) {

	const char* fp = env->GetStringUTFChars(file_path, 0);
	//println(fp);
	io::InputBuffer* ib = createInputBuffer(fp);
	std::string value;

	int total = 10000;
	jclass record_class = env->FindClass("bda/tensorflow/jni/input/Record");
	jmethodID constructor_record = env->GetMethodID(record_class, "<init>", "()V");
	jclass tensor_class = env->FindClass("bda/tensorflow/jni/Tensor");
	jmethodID constructor_tensor = env->GetMethodID(tensor_class, "<init>", "()V");
	jfieldID tensor_field = env->GetFieldID(record_class, "tensor", "Lbda/tensorflow/jni/Tensor;");
	jfieldID label_field = env->GetFieldID(record_class, "label", "I");
	jobjectArray record_array = env->NewObjectArray(total, record_class, NULL);
	int index = 0;

	while(true){
		value.clear();
		Status status = ib->ReadNBytes(1024 * 3 + 1, &value);
		if (status.ok()) {
			//std::cout << "value's size is " << value.size() << ", value's content is " << value << std::endl;
			//std::copy_n(value.data(), 1, lables_batch.flat<DataType::DT_INT64>().data());
			Tensor* tensor = new Tensor(DataType::DT_FLOAT, { 3, 32, 32 });
			unsigned char label = *value.data();
			float image[1024 * 3];
			for (int j = 0; j < 1024 * 3; j++) {
				unsigned char bt = *(value.data() + j + 1); //value.data() is a const char*
				image[j] = bt;
			}
			//lables_batch.flat<int64>()(lables_offset++) = label;	
			std::copy_n(image, 1024 * 3, (float*)tensor->flat<float>().data());
			jobject jtensor = env->NewObject(tensor_class, constructor_tensor);
			SetNativeAddress(env, jtensor, tensor);
			jobject jrecord = env->NewObject(record_class, constructor_record);
			env->SetObjectField(jrecord, tensor_field, jtensor);
			env->SetIntField(jrecord, label_field, label);

			env->SetObjectArrayElement(record_array, index, jrecord);
			// test first , maybe should be removed
			env->DeleteLocalRef(jtensor);
			env->DeleteLocalRef(jrecord);
			index++;
		}
		else {
			std::cout << "there is an error, error code is " << status.code() << ", error_message is " << status.error_message() << std::endl;
			break;
		}
	}
	delete ib;
	env->ReleaseStringUTFChars(file_path, fp);
	return record_array;
}

/*
* Class:     tensorflow_jni_input_Input
* Method:    sample
* Signature: ([Ltensorflow/jni/input/Record;[I)[Ltensorflow/jni/Tensor;
*/
JNIEXPORT jobjectArray JNICALL Java_bda_tensorflow_jni_input_Input_sample
(JNIEnv * env, jclass cls, jobjectArray jrecords, jintArray jindex) {
	//first get the dtype
	//println("sample have been invoked");
	jclass record_class = env->FindClass("bda/tensorflow/jni/input/Record");
	jfieldID tensor_field = env->GetFieldID(record_class, "tensor", "Lbda/tensorflow/jni/Tensor;");
	jfieldID label_field = env->GetFieldID(record_class, "label", "I");
	jobject record0 = env->GetObjectArrayElement(jrecords, 0);
	jobject tensor0 = env->GetObjectField(record0, tensor_field);
	Tensor * t0 = (Tensor*)GetNativeAddress(env, tensor0);
	DataType dt = t0->dtype();
	jint batch_size = env->GetArrayLength(jindex);
	Tensor * images = new Tensor(dt, { batch_size, 3, 32, 32 });
	Tensor * labels = new Tensor(DataType::DT_INT64, { batch_size });
	//println("before copy");
	//
	int start = 0;
	jint* index = env->GetIntArrayElements(jindex, 0);
	for (int i = 0; i < batch_size; i++) {
		jint id = index[i];
		jobject record = env->GetObjectArrayElement(jrecords, id);
		jobject tensor_i = env->GetObjectField(record, tensor_field);
		jint label_i = env->GetIntField(record, label_field);
		Tensor * t_i = (Tensor*)GetNativeAddress(env, tensor_i);
		//copy the images
		std::copy_n(t_i->flat<float>().data(), 1024 * 3, images->flat<float>().data() + i * 1024 * 3);
		//copy the lables
		labels->flat<int64>()(i) = label_i;
		//std::cout << labels->flat<int64>()(i) << std::endl;
		env->DeleteLocalRef(tensor_i);
		env->DeleteLocalRef(record);
	}
	env->ReleaseIntArrayElements(jindex, index, 0);
	//println("have copied the tensors into a batch tensor");
	jclass tensor_class = env->FindClass("bda/tensorflow/jni/Tensor");
	jmethodID constructor_tensor = env->GetMethodID(tensor_class, "<init>", "()V");
	jobject image_tensor = env->NewObject(tensor_class, constructor_tensor);
	SetNativeAddress(env, image_tensor, images);
	jobject labels_tensor = env->NewObject(tensor_class, constructor_tensor);
	SetNativeAddress(env, labels_tensor, labels);
	jobjectArray ret = env->NewObjectArray(2, tensor_class, NULL);
	env->SetObjectArrayElement(ret, 0, image_tensor);
	env->SetObjectArrayElement(ret, 1, labels_tensor);
	return ret;
}

