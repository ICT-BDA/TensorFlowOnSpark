package bda.common.util

import scala.collection.mutable.ListBuffer

/**
  * A key-value message class
  */
class Msg(kvs: ListBuffer[(String, Any)]) {

  def this() = this(ListBuffer.empty[(String, Any)])

  /** Append a key-value pair into message queue */
  def append(key: String, value: Any): this.type = {
    kvs.append((key, value))
    this
  }

  /** Return a string with format: key=value, ... */
  override def toString: String = kvs.map {
    case (k, v) => s"$k=$v"
  }.mkString(", ")
}

/** Factory method for Msg class */
object Msg {

  def apply(kvs: (String, Any)*): Msg = {
    val list: ListBuffer[(String, Any)] = kvs.to[ListBuffer]
    new Msg(list)
  }
}