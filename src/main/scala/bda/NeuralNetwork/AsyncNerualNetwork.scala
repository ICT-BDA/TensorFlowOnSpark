package bda.NeuralNetwork

import bda.common.obj.SeriesFeaturePoint
import bda.common.util.{Msg, Timer}
import bda.tensorflow.jni.{Tensor, TensorShape}
import bda.tensorflow.nn.network.{NetworkConfig, NetworkUtil}
import bda.tensorflow.run.{InitValueConfig, Item, VariableInitValue}
import bda.tensorflow.run.RNN.RNNMeta
import bda.tensorflow.util.Type
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import bda.common.Logging
import bda.tensorflow.jni_11.Input

import scala.collection.mutable.{ArrayBuffer, ListBuffer, Map}
import scala.collection.JavaConversions._

/**
  * Created by yixuanhe on 22/09/2016.
  */
object AsyncNeuralNetwork extends Logging {
  def train(train: RDD[SeriesFeaturePoint],
            valid: RDD[SeriesFeaturePoint] = null,
            config: NetworkConfig,
            iterNum: Int,
            xShape: List[Int],
            lr: Float,
            sc: SparkContext) = AsyncNeuralNetworkModel.train(train, valid, config, iterNum, xShape, lr, sc)

  def init(xShape: List[Int], sc: SparkContext, meta: RNNMeta, maxLen: Int): Array[Array[Float]] =
    meta.items.map(item => VariableInitValue.getValueFromConfig(item.config)).toArray
}

class AsyncNeuralNetworkModel(val weight: Array[Array[Float]],
                              val config: NetworkConfig,
                              val sc: SparkContext) extends Logging {
  def save(model_pt: String): Unit = {
    sc.parallelize(Seq(config, weight)).saveAsObjectFile(model_pt)
  }

  def predict(data: RDD[SeriesFeaturePoint], batchSize: Int): RDD[Array[Float]] = {
    System.loadLibrary("jt")

    var i = 0
    var points = new ListBuffer[SeriesFeaturePoint]


    val meta = new RNNMeta
    NetworkUtil.create(0, config, meta)

    val isRNN = config.isRNN
    val inputType = config.inputType
    val outputType = config.outputType
    val batchSize = config.batchSize
    val maxLen = config.inputSize
    val xShape = config.inputShape.toList

    val w = weight
    val c = config

    data.mapPartitionsWithIndex((index, iter) => {
      val result = new ListBuffer[Array[Float]]
      while (iter.hasNext) {
        points += iter.next()
        if (i % batchSize == 0) {
          val pair = AsyncNeuralNetworkTrainer.getData(points.toArray, maxLen, inputType, outputType, xShape, c, index)
          val predict = NetworkUtil.run(index, pair._2, pair._1, w, NetworkUtil.getPredict(index, c), c)

          predict.foreach(p => result += p)
          points = new ListBuffer[SeriesFeaturePoint]
        }
        i += 1
      }
      result.iterator
    })
  }

}

object AsyncNeuralNetworkModel {
  def apply(config: NetworkConfig, weight: Array[Array[Float]], maxLen: Int,
            xShape: List[Int], sc: SparkContext): AsyncNeuralNetworkModel =
    new AsyncNeuralNetworkModel(weight, config, sc)

  def apply(model: AsyncNeuralNetworkModel, weight: Array[Array[Float]]): AsyncNeuralNetworkModel =
    new AsyncNeuralNetworkModel(weight, model.config, model.sc)

  def train(train: RDD[SeriesFeaturePoint],
            valid: RDD[SeriesFeaturePoint],
            config: NetworkConfig,
            iterNum: Int,
            xShape: List[Int],
            lr: Float,
            sc: SparkContext) = AsyncNeuralNetworkTrainer.train(train, valid, config, iterNum, xShape, lr, sc)

  def load(sc: SparkContext, model_pt: String): AsyncNeuralNetworkModel = {
    val arr = sc.objectFile[Any](model_pt).collect()
    val config = arr(0).asInstanceOf[NetworkConfig]
    val weight = arr(1).asInstanceOf[Array[Array[Float]]]
    new AsyncNeuralNetworkModel(weight, config, sc)
  }
}

object AsyncNeuralNetworkTrainer extends Logging {
  System.loadLibrary("jt")

  def train(train: RDD[SeriesFeaturePoint],
            valid: RDD[SeriesFeaturePoint],
            config: NetworkConfig,
            iterNum: Int,
            xShape: List[Int],
            lr: Float,
            sc: SparkContext) = {
    val maxLen = train.map(points => points.data.length).reduce(_ max _)
    train.persist()
    if (valid != null)
      valid.persist()

    val train_data = train.mapPartitions(iter => Array(iter.toList).iterator)

    config.ps = System.getenv("TENSORFLOW_PS").split(",") //.map(p => "/job:"+p+"/task:0")
    config.master = System.getenv("TENSORFLOW_MASTER").split(",").map(p => "grpc://" + p)

    config.appId = sc.applicationId

    config.setInputSize(maxLen)
    config.setInputShape(xShape.toArray)
    // we need to get meta info, so don't delete it if you change code to get it!!!
    val meta = new RNNMeta
    config.async = true
    NetworkUtil.create(0, config, meta)

    val isRNN = config.isRNN
    val inputType = config.inputType
    val outputType = config.outputType
    val batchSize = config.batchSize

    if (config.outputSize == -1)
      config.outputSize = train.map(points =>
        points.data.map(point => point.label).max).reduce(_ max _).toInt

    var weight = AsyncNeuralNetwork.init(xShape, sc, meta, maxLen)
    val model = AsyncNeuralNetworkModel(config, weight, maxLen, xShape, sc)

    train.mapPartitionsWithIndex((index, iter) => {
      NetworkUtil.assignVariable(index, config, weight)
      Array(0).iterator
    }).foreach(a => Unit)

    val partitions = train_data.partitions.length

    val times = new ArrayBuffer[Double]()

    val out = train_data.mapPartitionsWithIndex((index, iter) => {
      val points = iter.next()
      for (i <- 0 until iterNum) {
        //TODO:we use a very ugly implement to extract batchSize point from iter, change it if has better way
        val timer = new Timer()
        val begin = i * batchSize % points.length
        val end = if (begin + batchSize > points.length) points.length else begin + batchSize

        val pair = getData(points.slice(begin, end).toArray, maxLen, inputType, outputType, xShape, config, index)
        val output = NetworkUtil.getApply(index, config) ++ NetworkUtil.getVariabe(index, config) :+ NetworkUtil.getAccuracy(index, config) :+ NetworkUtil.getLoss(index, config)

        val rate = new Tensor(Type.DT_FLOAT, new TensorShape(Array[Int]()))
        rate.initFromFloatArray(Array(lr))
        val inputs = pair._2 :+ NetworkUtil.getLr(index, config)
        val tensors = pair._1 :+ rate
        val outputs = NetworkUtil.run(index, inputs, tensors, output, config)

        val cost_time = timer.cost()
        val len = outputs.length

        println(i + " " + index + " accuracy: " + outputs(len - 2)(0) + " error:" + outputs(len - 1)(0) + " time cost :" + cost_time)

        if ((i % 10 == 0) && ((i / 10) % partitions == index)) {
          val update = outputs.slice(len/2-1, len-2)
          NetworkUtil.writeToHDFS(update, "/data/rnn/" + i + "_async.model")
        }
      }
      Array(0).iterator
    })
    out.foreach(a => Unit)


    logInfo(s"average iteration time: " + times.sum / times.size + "ms")

    val update = NetworkUtil.run(0, new Array[Input](0), new Array[Tensor](0), NetworkUtil.getVariabe(0, config), config)

    AsyncNeuralNetworkModel(model, update)
  }

  def getData(data: Array[SeriesFeaturePoint], maxLen: Int, xType: Int, yType: Int,
              xShape: List[Int], config: NetworkConfig, index: Int): (Array[Tensor], Array[Input]) = {
    val batchSize = data.length

    val inputLen = maxLen * 2 + 1

    val tensors: Array[Tensor] = new Array[Tensor](inputLen)
    val inputs: Array[Input] = new Array[Input](inputLen)
    val seq: Array[Float] = new Array[Float](batchSize * maxLen)

    val xshape = (batchSize :: xShape).toArray
    for (i <- 0 until maxLen) {
      val x: Tensor = new Tensor(xType, new TensorShape(xshape))
      val size = xshape.product
      val input: Array[Float] = new Array[Float](size)
      var begin = 0
      val iter_size = size / batchSize
      for (j <- 0 until batchSize) {
        val array = if (i < data(j).data.length) data(j).data(i).fs else new Array[Float](iter_size)
        Array.copy(array, 0, input, begin, iter_size)
        begin += iter_size
        seq(j * maxLen + i) = if (i < data(j).data.length) 1 else 0
      }
      x.initFromFloatArray(input)
      tensors(i) = x
      inputs(i) = NetworkUtil.getData(index, i, config)
    }

    val yshape = Array(batchSize)

    for (i <- 0 until maxLen) {
      val y = new Tensor(yType, new TensorShape(yshape))
      val size = batchSize
      //TODO : what if more type?
      if (yType == Type.DT_FLOAT) {
        val input = new Array[Float](size)
        for (j <- 0 until batchSize) {
          val value = if (i < data(j).data.length) data(j).data(i).label else 0
          input(j) = value.toFloat
        }
        y.initFromFloatArray(input)
      } else {
        val input = new Array[Int](size)
        for (j <- 0 until batchSize) {
          val value = if (i < data(j).data.length) data(j).data(i).label else 0
          input(j) = value.toInt
        }
        y.initFromIntArray(input)
      }
      tensors(i + maxLen) = y
      inputs(i + maxLen) = NetworkUtil.getLabel(index, i, config)
    }

    val s = new Tensor(Type.DT_FLOAT, new TensorShape(Array[Int](maxLen, batchSize)))
    s.initFromFloatArray(seq)
    tensors(maxLen * 2) = s
    inputs(maxLen * 2) = NetworkUtil.getSeq(index, config)

    (tensors, inputs)
  }
}

