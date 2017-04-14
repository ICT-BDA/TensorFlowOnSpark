package bda.NeuralNetwork

import java.util

import bda.common.obj.SeriesFeaturePoint
import bda.common.util.{Msg, Timer}
import bda.tensorflow.jni.{Tensor, TensorShape}
import bda.tensorflow.nn.Config
import bda.tensorflow.nn.network.{NetworkConfig, NetworkUtil}
import bda.tensorflow.run.{InitValueConfig, Item, VariableInitValue}
import bda.tensorflow.run.RNN.RNNMeta
import bda.tensorflow.util.Type
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import bda.common.Logging
import bda.tensorflow.jni_11.Input

import scala.collection.mutable.{ArrayBuffer, ListBuffer, Map}
import collection.JavaConversions._
import scala.collection.mutable
import scala.collection.JavaConversions._

/**
  * Created by yixuanhe on 22/09/2016.
  */
object NeuralNetwork extends Logging {
  def train(train: RDD[SeriesFeaturePoint],
            valid: RDD[SeriesFeaturePoint] = null,
            config: NetworkConfig,
            iterNum: Int,
            xShape: List[Int],
            lr: Float,
            sc: SparkContext,
            model_parallel: Boolean) = NeuralNetworkModel.train(train, valid, config, iterNum, xShape, lr, sc, model_parallel)

  def init(xShape: List[Int], sc: SparkContext, meta: RNNMeta, maxLen: Int): Array[Array[Float]] =
    meta.items.map(item => VariableInitValue.getValueFromConfig(item.config)).toArray
}

class NeuralNetworkModel(val weight: Broadcast[Array[Array[Float]]],
                         val config: NetworkConfig,
                         val sc: SparkContext) extends Logging {
  def save(model_pt: String): Unit = {
    sc.parallelize(Seq(config, weight.value)).saveAsObjectFile(model_pt)
  }

  def predict(data: RDD[SeriesFeaturePoint], batchSize: Int): Double = {
    System.loadLibrary("jt")

    var i = 1
    var points = new ListBuffer[SeriesFeaturePoint]


    val meta = new RNNMeta
    NetworkUtil.create(0, config, meta)

    val isRNN = config.isRNN
    val inputType = config.inputType
    val outputType = config.outputType
//    val batchSize = config.batchSize
    val maxLen = config.inputSize
    val xShape = config.inputShape.toList
    val predictName = meta.predictName

    val c = config
    val w = weight.value

    val result = data.mapPartitionsWithIndex((index, iter) => {
//      val result = new ListBuffer[Array[Float]]
      println("index : " + index)
      var p = 0.0
      val tmp = iter.toArray
      println(index, tmp.length)
      var pair = NeuralNetworkTrainer.getData(tmp.slice(0, 500), maxLen, inputType, outputType, xShape, c, index)
      var predict = NetworkUtil.run(index, pair._2, pair._1, w, Array(NetworkUtil.getAccuracy(index, c)), c)
      p += predict(0)(0) * 500
      println(predict(0)(0), 500,p)

      if (tmp.length > 500) {
        pair = NeuralNetworkTrainer.getData(tmp.slice(500, tmp.length), maxLen, inputType, outputType, xShape, c, index)
        predict = NetworkUtil.run(index, pair._2, pair._1, w, Array(NetworkUtil.getAccuracy(index, c)), c)
        p += predict(0)(0) * (tmp.length - 500)
        println(predict(0)(0), (tmp.length - 500), p)
      }
      Array(p).iterator
    })

    return result.reduce((a, b) => a + b)
  }

}

object NeuralNetworkModel {
  def apply(config: NetworkConfig, weight: Broadcast[Array[Array[Float]]], maxLen: Int,
            xShape: List[Int], sc: SparkContext): NeuralNetworkModel =
    new NeuralNetworkModel(weight, config, sc)

  def apply(model: NeuralNetworkModel, weight: Broadcast[Array[Array[Float]]]): NeuralNetworkModel =
    new NeuralNetworkModel(weight, model.config, model.sc)

  def train(train: RDD[SeriesFeaturePoint],
            valid: RDD[SeriesFeaturePoint],
            config: NetworkConfig,
            iterNum: Int,
            xShape: List[Int],
            lr: Float,
            sc: SparkContext,
            model_parallel: Boolean) = NeuralNetworkTrainer.train(train, valid, config, iterNum, xShape, lr, sc, model_parallel)

  def load(sc: SparkContext, model_pt: String): NeuralNetworkModel = {
    val arr = sc.objectFile[Any](model_pt).collect()
    val config = arr(0).asInstanceOf[NetworkConfig]
    val weight = sc.broadcast(arr(1).asInstanceOf[Array[Array[Float]]])
    new NeuralNetworkModel(weight, config, sc)
  }
}

object NeuralNetworkTrainer extends Logging {
  System.loadLibrary("jt")

  def train(train: RDD[SeriesFeaturePoint],
            valid: RDD[SeriesFeaturePoint],
            config: NetworkConfig,
            iterNum: Int,
            xShape: List[Int],
            lr: Float,
            sc: SparkContext,
            model_parallel: Boolean) = {
    val maxLen = train.map(points => points.data.length).reduce(_ max _)
    train.persist()
    if (valid != null)
      valid.persist()

    if (model_parallel) {
      config.ps = System.getenv("TENSORFLOW_PS").split(",").map(p => "grpc://" + p)
      config.master = System.getenv("TENSORFLOW_MASTER").split(",").map(p => "grpc://" + p)
    }

    config.appId = sc.applicationId

    config.setInputSize(maxLen)
    config.setInputShape(xShape.toArray)
    // we need to get meta info, so don't delete it if you change code to get it!!!
    val meta = new RNNMeta
    NetworkUtil.create(0, config, meta)

    val isRNN = config.isRNN
    val inputType = config.inputType
    val outputType = config.outputType
    val batchSize = config.batchSize

    if (config.outputSize == -1)
      config.outputSize = train.map(points =>
        points.data.map(point => point.label).max).reduce(_ max _).toInt

    var weight = NeuralNetwork.init(xShape, sc, meta, maxLen)
    var model = NeuralNetworkModel(config, sc.broadcast(weight), maxLen, xShape, sc)
    val partitions = train.partitions.length

    println("total partitions : " + partitions)

    val times = new ArrayBuffer[Double]()
    for (i <- 0 until iterNum) {
      val timer = new Timer()
      weight = model.weight.value
      val error = train.mapPartitionsWithIndex((index, iter) => {
        //TODO:we use a very ugly implement to extract batchSize point from iter, change it if has better way
        val points = iter.toList
        val begin = i * batchSize % points.length
        val end = if (begin + batchSize > points.length) points.length else begin + batchSize

        val pair = getData(points.slice(begin, end).toArray, maxLen, inputType, outputType, xShape, config, index)
        val output = NetworkUtil.getGradient(index, config) :+ NetworkUtil.getAccuracy(index, config) :+ NetworkUtil.getLoss(index, config)
        var errors = NetworkUtil.run(index, pair._2, pair._1, weight, output, config)

        val len = errors.length
        errors(len - 2)(0) = errors(len - 2)(0) * (end - begin + 1)

        Array(errors).iterator
      }).reduce((arr1, arr2) => {
        arr1.zip(arr2).map(a => a._1.zip(a._2).map(f => f._1 + f._2))
      })

      val len = error.length
      val accuracy = error(len - 2)(0)
      val loss = error(len - 1)(0)

      //      val rate = lr / (math.sqrt(i + 1) * partitions)
      val rate = lr / partitions
      val cost_time = timer.cost()
      val msg = Msg("iter" -> i,
        "time cost(ms)" -> cost_time,
        "accuracy(train)" -> accuracy,
        "loss" -> loss)

      val update = weight.zip(error.slice(0, len - 2)).map(a => a._1.zip(a._2).map(f => (f._1 - rate * f._2).toFloat))

      //      if (valid != null) {
      //        val errors = valid.mapPartitionsWithIndex((index, iter) => {
      //          val points = iter.toList.toArray
      //          val pair = getData(points, maxLen, inputType, outputType, xShape, config, index)
      //          val errors = NetworkUtil.run(index, pair._2, pair._1, weight, Array(NetworkUtil.getAccuracy(index, config)), config)
      //          Array(errors).iterator
      //        }).reduce((arr1, arr2) => {
      //          arr1.zip(arr2).map(a => a._1.zip(a._2).map(f => f._1 + f._2))
      //        })
      //        msg.append("accuracy(validate)", errors(0)(0) / partitions)
      //      }

      logInfo(msg.toString)
      times.append(cost_time)

      model = NeuralNetworkModel(model, sc.broadcast(update))
      if (i % 10 == 0)
        model.save("hdfs://10.61.1.119:9000/data/rnn/" + i + "_sync.model")
    }
    logInfo(s"average iteration time: " + times.sum / times.size + "ms")
    model
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

