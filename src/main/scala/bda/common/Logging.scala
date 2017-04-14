package bda.common

import org.slf4j.{Logger, LoggerFactory}

/**
 * Utility trait for Logging
 * Usage:
 * {{{
 * class A extends Logging {
 *    ...
 *    logDebug(msg)
 *    logInfo(msg)
 *    logWarn(msg)
 *    logError(msg)
 * }
 * }}}
 *
 * The log level can be set in the file "log4j.properties" in package bda.
 */
trait Logging {
  // Make the log field transient so that objects with Logging can
  // be serialized and used on another machine
  @transient private var log_ : Logger = null

  /** Method to get or create the logger for this object */
  protected def log: Logger = {
    if (log_ == null) {
      var className = this.getClass.getName
      // Ignore trailing $'s in the class names for Scala objects
      if (className.endsWith("$")) {
        className = className.substring(0, className.length - 1)
      }
      log_ = LoggerFactory.getLogger(className)
    }
    log_
  }

  /** log a information message */
  protected def logInfo(msg: => String) {
    if (log.isInfoEnabled) log.info(msg)
  }

  /** Log a debug message */
  protected def logDebug(msg: => String) {
    if (log.isDebugEnabled) log.debug(msg)
  }

  protected def logTrace(msg: => String) {
    if (log.isTraceEnabled) log.trace(msg)
  }

  /** Log a warining message */
  protected def logWarn(msg: => String) {
    if (log.isWarnEnabled) log.warn(msg)
  }

  /** Log a error message, and terminate the program. */
  protected def logError(msg: => String): Null = {
    if (log.isErrorEnabled) log.error(msg)
    null
  }

  /** Log methods that take Throwables (Exceptions/Errors) too */
  protected def logInfo(msg: => String, throwable: Throwable) {
    if (log.isInfoEnabled) log.info(msg, throwable)
  }

  /** Log a debug message, and take Throwables (Exceptions/Errors) too */
  protected def logDebug(msg: => String, throwable: Throwable) {
    if (log.isDebugEnabled) log.debug(msg, throwable)
  }

  /** Log a trace message, and take Throwables (Exceptions/Errors) too */
  protected def logTrace(msg: => String, throwable: Throwable) {
    if (log.isTraceEnabled) log.trace(msg, throwable)
  }

  /** Log a warning message, and take Throwables (Exceptions/Errors) too */
  protected def logWarn(msg: => String, throwable: Throwable) {
    if (log.isWarnEnabled) log.warn(msg, throwable)
  }

  /** Log a error message, and take Throwables (Exceptions/Errors) too. */
  protected def logError(msg: => String, throwable: Throwable) {
    if (log.isErrorEnabled) log.error(msg, throwable)
  }
}
