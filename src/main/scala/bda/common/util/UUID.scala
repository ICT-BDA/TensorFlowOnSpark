package bda.common.util

/**
  *  unique ID Generator
  */
object UUID {

  def rand(prefix: String = ""): String =
    prefix + "_" + java.util.UUID.randomUUID().toString.takeRight(12)
}
