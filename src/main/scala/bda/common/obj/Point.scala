package bda.common.obj
/**
  * Point with Dense features
  */
case class LabeledDensePoint(id: String,
                             label: Double,
                             fs: Array[Float]) {
  override def toString =
    s"$id\t$label\t${fs.length}"
}

object LabeledDensePoint {
  val default_id = "0"
  val default_label = 0.0

  def apply(label: Double, fs: Array[Float]): LabeledDensePoint =
    new LabeledDensePoint(default_id, label, fs)

  def apply(fs: Array[Float]): LabeledDensePoint =
    new LabeledDensePoint(default_id, default_label, fs)

  def apply(id: String, fs: Array[Float]): LabeledDensePoint =
    new LabeledDensePoint(id, default_label, fs)

  /**
    * parse from string
    *
    * @param s format: label  v,v,v,v,v
    * @return
    */
  def parse(s: String): LabeledDensePoint = s.split("\t") match {
    case Array(fs) =>
      val feature = fs.split(",").map(f => f.toFloat)
      LabeledDensePoint(feature)
    case Array(label, fs) =>
      val feature = fs.split(",").map(f => f.toFloat)
      LabeledDensePoint(label.toDouble, feature)
  }
}

case class SeriesFeaturePoint(id: String,
                              data: Array[LabeledDensePoint]) {
  override def toString =
    s"$id\t${data.length}"
}

/**
  * Point store series data, used for nerual network
  */
object SeriesFeaturePoint {
  val default_id = "0"

  def apply(data: Array[LabeledDensePoint]): SeriesFeaturePoint =
    new SeriesFeaturePoint(default_id, data)

  def parse(s: String): SeriesFeaturePoint = {
    val series = s.split(" ").map(f => LabeledDensePoint.parse(f))
    SeriesFeaturePoint(series)
  }
}