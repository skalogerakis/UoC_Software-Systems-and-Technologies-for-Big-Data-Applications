package org.example

import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Success, Try}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object codeSnippet {

  def main(args: Array[String]) {
    //As the link below states, spark-shell creates by default Spark Context so use that
    //    https://sparkbyexamples.com/spark/sparksession-vs-sparkcontext/?fbclid=IwAR15guKOla8APJa3paaFCNbmfkRRhVp_Il_tOo9F005XpECpj2m1R-uGXkU
    val spark = new SparkContext(new SparkConf().setAppName("codeSnippets").setMaster("local[*]"))

    val baseRdd = spark.textFile("/home/skalogerakis/Projects/CS543/CS543-Assignment2/hw2/dataset.csv")

    println("============1.2===============")
    println("Sanity check count for data points(baseRDD): " + baseRdd.count())
    println("1.2.2 - Top 5 datapoints")
    baseRdd.take(5).foreach(x => println(x))

    println("============1.3===============")
    //Keep everything in cache with cache, as we will use this again and again
//    val parsePointsRdd = baseRdd.foreach(x => inputToLabeled_1_3(x))

    val parsePointsRdd = baseRdd.map(x=>inputToLabeled_1_3(x)).persist()

    println("============1.3.3===============")
    println("Label of the first element: "+parsePointsRdd.first().label)

    println("============1.3.4===============")
    println("Label of the first element: "+parsePointsRdd.first().features)

    println("============1.3.5===============")
    println("Length of the features of the first element: "+parsePointsRdd.first().features.size)

    println("============1.3.6===============")
    println("MAX: "+parsePointsRdd.map(x => x.label).max)
    val minLabel = parsePointsRdd.map(x => x.label).min
    println("MIN: "+minLabel)

    //IMPORTANT: Return again RDD of LabelPoint
    val shiftedPointsRdd = parsePointsRdd.map(x => LabeledPoint(x.label-minLabel,x.features))

    //Debugging
    shiftedPointsRdd.foreach(println)

    println("============1.4===============")
    println("MAX: "+shiftedPointsRdd.map(x => x.label).max)
    println("MIN: "+shiftedPointsRdd.map(x => x.label).min)

    println("============1.5===============")
    val weights = Array(.8, .1, .1)
    val seed = 42
    val Array(trainData, valData, testData) = shiftedPointsRdd.randomSplit(weights, seed)

    //As 1.5.2 requests, keep all three tables in memory
    trainData.persist()
    valData.persist()
    testData.persist()

    println("============1.5.3===============")
    println("TrainData count: "+trainData.count())
    println("valData count: "+valData.count())
    println("testData count: "+testData.count())
    println("TOTAL count: "+(testData.count()+trainData.count()+valData.count()))
  }

  def inputToLabeled_1_3(str: String): LabeledPoint ={

    //Could simply map(s=>s.toDouble) as we know that our dataset contains valid double values. To make everything bulletproof apply the idea in the link below.
    // Check if our value can become double and only then convert it to double. Avoid exceptions
    //    https://stackoverflow.com/questions/33848755/convert-a-scala-list-of-strings-into-a-list-of-doubles-while-discarding-unconver
    val attrList = str.split(",").map( s => Try(s.toDouble) ).collect { case Success(x) => x }.toList
//    val attrList = str.split(",").map( s => s.toDouble).toList

//    println(attrList)

    //Label is the first value(using head list function) and features are the rest(tail function takes all but first element)
    LabeledPoint(attrList.head, Vectors.dense(attrList.tail.toArray))

  }


}
