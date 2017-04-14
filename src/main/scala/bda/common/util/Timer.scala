package bda.common.util

/**
 * A Timer class records a
 * Start time is the create time of the object, if start() is not execute.
 */
class Timer {
  /** Start time to track */
  private var start_t = System.nanoTime()

  /** Reset the starting time to now */
  def reset(): this.type = {
    start_t = System.nanoTime()
    this
  }

  /** Return the millisecond pasted from start time */
  def cost(): Long = {
    val t = System.nanoTime()
    (t - start_t) / 1e6.toLong
  }

  /** A string representation of millisecond cost */
  override def toString: String = s"cost: $cost millisecond ms"
}