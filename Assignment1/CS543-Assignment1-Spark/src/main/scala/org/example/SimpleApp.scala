package org.example

import org.apache.hadoop.yarn.util.RackResolver
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.matching.Regex
//import org.apache.spark.sql.SparkSession

case class Log (host: String, date: String, requestURI: String, status: Int, bytes: Int)
case class LogT (host: String, date: String, status: Int)

object SimpleApp {
  def main(args: Array[String]) {

    //As the link below states, spark-shell creates by default Spark Context so use that
//    https://sparkbyexamples.com/spark/sparksession-vs-sparkcontext/?fbclid=IwAR15guKOla8APJa3paaFCNbmfkRRhVp_Il_tOo9F005XpECpj2m1R-uGXkU
    val spark = new SparkContext(new SparkConf().setAppName("SimpleApp").setMaster("local[*]"))

    val baseRdd = spark.textFile("/home/skalogerakis/Projects/CS543/Assignment1/CS543-Assignment1-Spark/NASA_access_log_Jul95")
    println("Sanity check for baseRDD: "+baseRdd.count())
    //    val patternHost = """^([^\s]+\s)""".r
//    val patternTime = """^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]""".r
//    val patternRequest = """^.*"\w+\s+([^\s]+)\s*.*"""".r
//    val patternStatus = """^.*"\s+([^\s]+)""".r
//    val patternBytes = """^.*\s+(\d+)$""".r
//    val printRdd = baseRdd.map( x => getLogFields(x))
//    print(printRdd.count())

    //BadRDD finder
//    val badRDD = baseRdd.filter(x => patternBytes.findFirstIn(x) == None)
//    print(badRDD.count())
//    badRDD.take(15).foreach(println)

//    val badRdd = splitRDD.filter(x => findFirstIn(x) == None)
//    badRdd.count
//    badRdd.take(3)
//    print(splitRDD.count())


    val cleanRDD = convertToLog(baseRdd).persist()
    println("Sanity check for cleanRDD: "+cleanRDD.count())

    println("============PART 2.3===============")
    exploreContentSize_2_3_1(cleanRDD)
    HTTPAnalysis_2_3_2(cleanRDD)
    FrequencyHosts_2_3_3(cleanRDD)
    ErrorPaths_2_3_4(cleanRDD)
    UniqueHosts_2_3_5(cleanRDD)
    Error404Counter_2_3_6(cleanRDD)
    DistinctError404_2_3_7(cleanRDD)
    MostError404_2_3_8(cleanRDD)

  }

  def exploreContentSize_2_3_1(cleanRDD : RDD[Log]): Unit ={
    println("============2.3.1===============")
    val contentOnly = cleanRDD.map( x => x.bytes)

    println("MAX:" +contentOnly.max())
    println("MIN:" +contentOnly.min())
    println("AVG:" +contentOnly.sum() / contentOnly.count())
  }

  def HTTPAnalysis_2_3_2(cleanRDD : RDD[Log]): Unit ={
    println("============2.3.2===============")

    //ReduceByKey is better than groupByKey as states in the documentation, with less network transfers
    val statusOnly = cleanRDD.map( curVal => (curVal.status,1))
                              .reduceByKey((x,y) => x+y)


    //We want the 100 most frequent, so sort status first and then take the 100 elements
    statusOnly.sortBy(freq => freq._2,ascending = false)
              .take(100)
              .foreach(x => println("STATUS: "+x._1+" FREQ: "+x._2))
  }

  def FrequencyHosts_2_3_3(cleanRDD : RDD[Log]): Unit ={
    println("============2.3.3===============")

    //Similarly to the previous case, but also filter only the results with more than 10 entries
    val hosts = cleanRDD.map( curVal => (curVal.host,1))
                        .reduceByKey((x,y) => x+y)
                        .persist()    //EDIT: Keep that in the cache we use it 2_3_5



    hosts.filter(f => f._2 > 10).take(10).foreach(pr => println("HOSTS :"+pr._1+" FREQ: "+pr._2))

  }

  def ErrorPaths_2_3_4(cleanRDD : RDD[Log]): Unit ={
    println("============2.3.4===============")

    //Filter before the first map, it is more efficient that way with less computations
    val errors = cleanRDD.filter( f => f.status != 200)
                          .map( curVal => (curVal.requestURI,1))
                          .reduceByKey((x,y) => x+y)
                          .sortBy(freq => freq._2,ascending = false)
                          .take(10)

    errors.foreach(pr => println("ERRORS :"+pr._1+" FREQ: "+pr._2))

  }

  def UniqueHosts_2_3_5(cleanRDD : RDD[Log]): Unit ={
    println("============2.3.5===============")

    //Filter before the first map, it is more efficient that way with less computations
    //INITIAL EFFORT: This solution works, but is not optimized as it shuffles through partition. Use reduceByKey instead
    //https://stackoverflow.com/questions/30959955/how-does-distinct-function-work-in-spark/31127175
//    val errors = cleanRDD.map( x => (x.host)).distinct().count()

    val errors = cleanRDD.map( curVal => (curVal.host,1))
                          .reduceByKey((x,y) => x+y)
                          .count()

    //We stored in cache this from the previous function, use it to count the distinct. Identical result with built-in distinct function, so everything works as expected
    println("Number of unique hosts: "+errors)

  }

  def Error404Counter_2_3_6(cleanRDD : RDD[Log]): Unit ={
    println("============2.3.6===============")

    val counter404 = cleanRDD.filter( f => f.status == 404 ).persist()  //Keep data in the cache, use it in 2.3.7


    //We stored in cache this from the previous function, use it to count the distinct. Identical result with built-in distinct function, so everything works as expected
    println("Number of 404 Response Codes: "+counter404.count())

  }

  def DistinctError404_2_3_7(cleanRDD : RDD[Log]): Unit ={
    println("============2.3.7===============")

    //As in 2.3.5 could use distint function, but reduceByKey is better with less shuffling
    val distinct404 = cleanRDD.filter( f => f.status == 404 )
                              .map( x=> (x.requestURI,1))
                              .reduceByKey((x,y) => x+y)
                              .persist()    //Persist to use in 2,3.8



    //We stored in cache this from the previous function, use it to count the distinct. Identical result with built-in distinct function, so everything works as expected
    distinct404.take(40).foreach(x => println(x._1))

  }

  def MostError404_2_3_8(cleanRDD : RDD[Log]): Unit ={
    println("============2.3.8===============")

    val distinct404 = cleanRDD.filter( f => f.status == 404 )
                              .map( x=> (x.requestURI,1))
                              .reduceByKey((x,y) => x+y)
                              .sortBy(freq => freq._2,ascending = false)
                              .take(20)

    distinct404.foreach(pr => println("URI :"+pr._1+" FREQ: "+pr._2))

  }

  //INIT LOG FUNCTION, FROM ASSIGNMENT
  def getLogFields(str: String):Log = {
    val patternHost = """^([^\s]+\s)""".r
    val patternTime = """^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]""".r
    val patternRequest = """^.*"\w+\s+([^\s]+)\s*.*"""".r
    val patternStatus = """^.*"\s+([^\s]+)""".r
    val patternBytes = """^.*\s+(\d+)$""".r

    Log (patternHost.findAllIn(str).matchData.next().group(1),
      patternTime.findAllIn(str).matchData.next().group(1),
      patternRequest.findAllIn(str).matchData.next().group(1),
      patternStatus.findAllIn(str).matchData.next().group(1).toInt,
      patternBytes.findAllIn(str).matchData.next().group(1).toInt
    )


  }

  //My LogField parser. Tackle corner cases, and make it robust to execute correctly for all files
  def getImprovedLogFields(str: String): Log = {
    val patternHost = """^([^\s]+\s)""".r
    val patternTime = """^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]""".r
    val patternRequest = """^.*"\w+\s+([^\s]+)\s*.*""".r
    val patternStatus = """^.*"\s+([^\s]+)""".r
    val patternBytes = """^.*\s+(\d+)$""".r

    /*
      All fields cause issues in the full dataset, some more than others.
        1. The most frequent case is Bytes(count: )
        2. Field requestURI also causes some exceptions due to encoding issues
        3. Host, Time, Status fields cause issues due to the last line
     */

    Log(if (patternHost.findFirstIn(str) == None) "" else patternHost.findAllIn(str).matchData.next().group(1),
      if (patternTime.findFirstIn(str) == None) "" else patternTime.findAllIn(str).matchData.next().group(1),
      if (patternRequest.findFirstIn(str) == None) "" else patternRequest.findAllIn(str).matchData.next().group(1),
      if (patternStatus.findFirstIn(str) == None) 0 else patternStatus.findAllIn(str).matchData.next().group(1).toInt,
      if (patternBytes.findFirstIn(str) == None) 0 else patternBytes.findAllIn(str).matchData.next().group(1).toInt
    )
  }

  //Passes baseRDD, apply the newly appended Log
  def convertToLog(base : RDD[String]): RDD[Log] = {
    val temp = base.map(x => getImprovedLogFields(x))
    return temp
  }


}


