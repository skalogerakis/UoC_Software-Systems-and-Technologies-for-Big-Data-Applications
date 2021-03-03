package org.example

import scala.util.matching.Regex
import org.apache.spark
import org.apache.spark.rdd.RDD
import org.example.SimpleApp.getLogFields

import scala.util.Sorting


object WebLogger {

  def main(args: Array[String]) {
    import org.apache.spark.sql.SparkSession

    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .master("local[*]")
      .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
//    val baseRdd = spark.read.textFile("/home/skalogerakis/Projects/CS543/Assignment1/CS543-Assignment1-Spark/NASA_access_log_Jul95")

//    val splitRDD = baseRdd.map(getLogFields).persist()

//    splitRDD.map( x => x.requestURI ).count()


    val baseRdd = spark.read.textFile("/home/skalogerakis/Projects/CS543/Assignment1/CS543-Assignment1-Spark/NASA_access_log_Jul95")
    baseRdd.map(getImprovedLogFields).map(x => x.requestURI  ).count

    // Create the cleanRdd finally with the new convertToLog function
//    val cleanRdd = convertToLog(baseRdd).cache
//    cleanRdd.count
//    // cleanRdd.take(1)
//
//    println("---------- < 2.3 > ----------")
//    println("1. Explore Content Size")
//    contentSize(cleanRdd)
//
//    println("2. HTTP Status Analysis : 100 most frequent Status Val")
//    httpStatusAnalysis(cleanRdd)
//
//    println("3. Frequent Hosts")
//    frequentHosts(cleanRdd)
//
//    println("4. Top 10 Error Paths")
//    top10errorPath(cleanRdd)
//
//    println("5. Unique Hosts")
//    getUniqueHosts(cleanRdd)
//
//    println("6. Number of 404 Response Codes")
//    count404(cleanRdd)
//
//    println("7. 40 Distinct URI's generating 404")
//    distinctUris404(cleanRdd)
//
//    println("8. Top 20 paths generating 404")
//    topTwentyPaths(cleanRdd)
  }

  case class Log(host: String, date: String, requestURI: String, status: Int, bytes: Int)

  def getLogFields(str: String): Log = {
    val patternHost = """^([^\s]+\s)""".r
    val patternTime = """^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]""".r
    val patternRequest = """^.*"\w+\s+([^\s]+)\s*.*""".r
    val patternStatus = """^.*"\s+([^\s]+)""".r
    val patternBytes = """^.*\s+(\d+)$""".r

    Log(patternHost.findAllIn(str).matchData.next().group(1),
      patternTime.findAllIn(str).matchData.next().group(1),
      patternRequest.findAllIn(str).matchData.next().group(1),
      patternStatus.findAllIn(str).matchData.next().group(1).toInt,
      patternBytes.findAllIn(str).matchData.next().group(1).toInt
    )
  }

  def getImprovedLogFields(str: String): Log = {
    val patternHost = """^([^\s]+\s)""".r
    val patternTime = """^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]""".r
    val patternRequest = """^.*"\w+\s+([^\s]+)\s*.*""".r
    val patternStatus = """^.*"\s+([^\s]+)""".r
    val patternBytes = """^.*\s+(\d+)$""".r

    val patternBytes_ = if (patternBytes.findFirstIn(str) == None) 0 else patternBytes.findAllIn(str).matchData.next().group(1).toInt
    val patternStatus_ = if (patternStatus.findFirstIn(str) == None) 0 else patternStatus.findAllIn(str).matchData.next().group(1).toInt
    val patternRequest_ = if (patternRequest.findFirstIn(str) == None) "" else patternRequest.findAllIn(str).matchData.next().group(1)
    val patternTime_ = if (patternTime.findFirstIn(str) == None) "" else patternTime.findAllIn(str).matchData.next().group(1)
    val patternHost_ = if (patternHost.findFirstIn(str) == None) "" else patternHost.findAllIn(str).matchData.next().group(1)

    Log(patternHost_,
      patternTime_,
      patternRequest_,
      patternStatus_,
      patternBytes_
    )
  }

  def convertToLog(base: RDD[String]): RDD[Log] = {
    val temp = base.map(getImprovedLogFields)
    temp
  }

  def contentSize(rdd: RDD[Log]) = {
    val max = rdd.map(x => (x.bytes)).max
    val min = rdd.map(x => (x.bytes)).min
    val average = rdd.map(x => (x.bytes)).sum / rdd.count


    println("Min = " + min)
    println("Max = " + max)
    println("Average = " + average)
  }

  def httpStatusAnalysis(rdd: RDD[Log]) = {
    rdd.map(x => (x.status, 1)).reduceByKey(_ + _).foreach(println)
  }

  def frequentHosts(rdd: RDD[Log]) = {
    rdd.map(x => (x.host, 1)).reduceByKey(_ + _).filter(x => x._2 > 10).take(10).foreach(println)

  }

  def top10errorPath(rdd: RDD[Log]) = {
    rdd.filter(x => x.status != 200).map(x => (x.requestURI, 1)).reduceByKey(_ + _).collect().toList.sortBy(-_._2).take(10).foreach(println)
  }

  def getUniqueHosts(rdd: RDD[Log]) = {
    rdd.map(x => (x.host, 1)).reduceByKey(_ + _).count
  }

  def count404(rdd: RDD[Log]) = {
    rdd.filter(x => x.status == 404).count
  }

  def distinctUris404(rdd: RDD[Log]) = {
    rdd.filter(x => x.status == 404).map(x => x.requestURI).distinct().take(40).foreach(println)
  }

  def topTwentyPaths(rdd: RDD[Log]) = {
    rdd.filter(x => x.status == 404).map(x => (x.requestURI, 1)).reduceByKey(_ + _).collect().toList.sortBy(-_._2).take(20).foreach(println)
  }


  // Here i load the nasa.txt file to the baseRdd


}