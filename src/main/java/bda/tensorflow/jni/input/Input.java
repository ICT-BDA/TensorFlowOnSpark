package bda.tensorflow.jni.input;

import bda.tensorflow.jni.Tensor;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class Input {
    public static native Record[] createImageAndLabelArrayFromFile(String paramString);

    public static Tensor[] sampleFromRecords(Record[] records, int number) {
        int length = records.length;
        int[] index = new int[number];
        Random rand = new Random();
        Set<Integer> set = new HashSet();
        for (int i = 0; i < number; i++) {
            int index_i = rand.nextInt(length);
            while (set.contains(Integer.valueOf(index_i))) {
                index_i = rand.nextInt(length);
            }
            set.add(Integer.valueOf(index_i));
            index[i] = index_i;
        }
        return sample(records, index);
    }

    public static Tensor[] sampleAllRecords(Record[] records) {
        int length = records.length;
        int[] index = new int[length];
        for (int i = 0; i < length; i++) {
            index[i] = i;
        }
        return sample(records, index);
    }

    private static native Tensor[] sample(Record[] paramArrayOfRecord, int[] paramArrayOfInt);

    public static native Record convertToRecord(byte paramByte, float[] paramArrayOfFloat);
}