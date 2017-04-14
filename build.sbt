
name := "TensorFlowOnSpark"

version := "1.0"

scalaVersion := "2.11.4"

libraryDependencies ++= Seq(
  "org.apache.commons" % "commons-math3" % "3.5",
  "org.apache.commons" % "commons-lang3" % "3.4",
  "org.apache.spark" %% "spark-core" % "1.5.2",
  "org.apache.spark" %% "spark-sql" % "1.5.2",
  "org.apache.spark" %% "spark-graphx" % "1.5.2",
  "com.github.scopt" %% "scopt" % "3.2.0",
  "org.scalatest" %% "scalatest" % "2.2.1" % "test",
  "com.fasterxml.jackson.core" % "jackson-core" % "2.8.3",
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.8.3",
  "com.fasterxml.jackson.core" % "jackson-annotations" % "2.8.3",
  "org.slf4j" % "slf4j-api" % "1.7.12",
  "org.slf4j" % "slf4j-log4j12" % "1.7.12",
  "org.scalatest" %% "scalatest" % "2.2.1" % "test",
  "com.github.nscala-time" %% "nscala-time" % "2.2.0"
)