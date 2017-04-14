package bda.NeuralNetwork

import java.util

import bda.common.Logging
import bda.common.obj.SeriesFeaturePoint
import bda.tensorflow.nn.network.{NetworkConfig, NetworkUtil}
import bda.tensorflow.run.{InitValueConfig, Item, VariableInitValue}
import bda.tensorflow.run.RNN.RNNMeta

import scala.collection.mutable.{ListBuffer, Map}
import collection.JavaConversions._
import scala.collection.mutable

/**
  * Created by yixuanhe on 22/09/2016.
  */
object LocalNeuralNetwork extends Logging {
  def train(train: Array[SeriesFeaturePoint],
            test: Array[SeriesFeaturePoint] = null) = {}

  def init(xShape: List[Int], meta: RNNMeta, maxLen: Int): Array[Array[Float]] =
    meta.items.map(item => VariableInitValue.getValueFromConfig(item.config)).toArray
}

class LocalNeuralNetworkModel(val weight: Array[Array[Float]],
                              val config: NetworkConfig) extends Logging {
  def save(path: String): Unit = {}

  def predict(data: Array[SeriesFeaturePoint], batchSize: Int): Array[Array[Float]] = {
    System.loadLibrary("jt")

    var i = 0
    val points = new ListBuffer[SeriesFeaturePoint]
    val result = new ListBuffer[Array[Float]]

    val iter = data.iterator

    val meta = new RNNMeta
    NetworkUtil.create(0, config, meta)

    val isRNN = config.isRNN
    val inputType = config.inputType
    val outputType = config.outputType
    val batchSize = config.batchSize
    val maxLen = config.inputSize
    val xShape = config.inputShape.toList
    val predictName = meta.predictName

    while (iter.hasNext) {
      points += iter.next()
      if (i % batchSize == 0) {
        val pair = NeuralNetworkTrainer.getData(points.toArray, maxLen, inputType, outputType, xShape, config, 0)
        val predict = NetworkUtil.run(0, pair._2, pair._1, weight, meta.predict, config)

        predict.foreach(p => result += p)
      }
      i += 1
    }
    result.toArray
  }
}

object LocalNeuralNetworkModel {
  def apply(config: NetworkConfig, weight: Array[Array[Float]], maxLen: Int,
            xShape: List[Int]): LocalNeuralNetworkModel =
    new LocalNeuralNetworkModel(weight, config)

  def apply(model: LocalNeuralNetworkModel, weight: Array[Array[Float]]): LocalNeuralNetworkModel =
    new LocalNeuralNetworkModel(weight, model.config)

  def train(data: Array[SeriesFeaturePoint]) = {}
}

object LocalNeuralNetworkTrainer {
  System.loadLibrary("jt")

  def train(train: Array[SeriesFeaturePoint],
            valid: Array[SeriesFeaturePoint],
            config: NetworkConfig,
            iterNum: Int,
            xShape: List[Int],
            lr: Float,
            parallel: Boolean = false) = {
    val maxLen = train.map(points => points.data.length).reduce(_ max _)
    config.setInputSize(maxLen)
    // we need to get meta info, so don't delete it if you change code to get it!!!
    val meta = new RNNMeta

    val isRNN = config.isRNN
    val inputType = config.inputType
    val outputType = config.outputType
    val batchSize = config.batchSize
    val lossName = meta.lossName

    val names = new util.HashSet[String]()
    names.addAll(meta.grad2name.keySet)
    names.add(meta.lossName)
    val length: Int = names.size
    val gradNames: Array[String] = names.toArray(new Array[String](length))


    if (config.outputSize == -1)
      config.outputSize = train.map(points =>
        points.data.map(point => point.label)
          .reduce(_ max _)).reduce(_ max _).toInt

    var weight = LocalNeuralNetwork.init(xShape, meta, maxLen)
    var model = LocalNeuralNetworkModel(config, weight, maxLen, xShape)

    for (i <- 0 until iterNum) {
      weight = model.weight

      val points = train
      val begin = i * batchSize % points.length
      val end = if (begin + batchSize > points.length) points.length else begin + batchSize

      val pair = NeuralNetworkTrainer.getData(points.slice(begin, end), maxLen, inputType, outputType, xShape, config, 0)
      val error = NetworkUtil.run(0, pair._2, pair._1, weight, meta.gradient, config)

      val rate = lr / math.sqrt(i + 1)
      val update = weight.zip(error).map(a => a._1.zip(a._2).map(f => (f._1 - rate * f._2).toFloat))
      model = LocalNeuralNetworkModel(model, update)
    }
  }
}
