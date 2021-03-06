import scala.annotation.tailrec

object ScalaWarmUp {

  def main(args: Array[String]): Unit = {
      println("Hello, World")
    val x = List(7,5,32,43,61,523)
    val y = List(1)
    val z = List()
    val text = List("c","c","b","a","s")
//    val text = List("c")

    val substringC = List("absc","abfdsg","asfxzcvf")

    val tmpTests = List(")",")","(","(")

    val tmpTest = List("s","(","(","a",")",")")
//    println(getSecondToLast1(x))
//
//    var b = getSecondToLast1(x) == x(x.size - 2)
//    println(b)

//    println("HEY "+getSecondToLast2(z))
      filterUnique(text).foreach(print)
//    print(getMostFrequentSubstring(substringC,2))
//    getMostFrequentSubstring(substringC,2)
//    print(checkBalancedParentheses(tmpTest))
  }

  //DONE
  //It is a good practise to use @tailrec for tail recursive function.
  @tailrec
  def getSecondToLast1(lst: List[Int]) : Int = {

          lst match {
            case secondLast :: _ :: Nil => secondLast // In this case we have only two remaining Elements so get the second to last
            case _ :: tail => getSecondToLast1(tail)  //Tail function returns all elements except first
            case _ => throw new NoSuchElementException
            /*
              Exception can occur in two cases:
                1. When list empty
                2. When list contains only one element
             */
          }


  }

  //DONE
  def getSecondToLast2(l: List[Int]): Int = {

    //As in the previous function, in case we have only 1 or two elements throw an exception and don't proceed any further
    if(l.size == 1 | l.size == 0){
      throw new NoSuchElementException
    }

    val listIndex = l.zipWithIndex  //Attach each list element an index using zipWithIndex ex (7,0)(5,1)(32,2)(43,3)(61,4)(523,5)

    val listFilter = listIndex.filter( t => t._2 == l.size - 2 )  //Scan and find the second to last entry, in this case (61,4)

    listFilter.map(_._1).head //Keep and return only the original list value. Head function will return the head of the list(Only one element in each case so it will return the correct one)

  }

  //DONE
  def filterUnique(l: List[String]): List[String] = {

    //In the case the list is empty throw error exception
    if(l.isEmpty){
      throw new NoSuchElementException
    }

    //Parse list from left to right. Function executes in the format (curElement,runningList). Start from an list containing the last element of the init list
    val tmp = l.foldRight[List[String]](List(l.last)){
      case (x,xs) if xs.head != x => x :: xs
      case (x,xs) => xs
      case _ => throw new NoSuchElementException
    }.toList

    return tmp

  }


  //DONE
  def getMostFrequentSubstring(lst: List[String], k: Int) = {

    //Create a list with all substrings of all strings, for a sliding window of k
    //We want only one total list, use flatmap for that reason and not map.(Could combine map and flatten which is identical)
      val allSubstr = {
        lst.flatMap{
          str => str.sliding(k)
        }
      }
//        println(allSubstr)

    //GroupMapReduce combines group, map and reduce function as its name state.
    //The idea is very simple, group by the entries, map with value 1 and add(reduce) those elements together to get the frequency
    //GroupMapReduce example/idea https://blog.genuine.com/2019/11/scalas-groupmap-and-groupmapreduce/

    val maxSubstringEntry = allSubstr
      .groupMapReduce(x => x)(_ => 1)(_ + _)
      .maxBy(_._2)

    println(maxSubstringEntry._1)
  }

  //DONE
  //balanced parentheses
  def checkBalancedParentheses(lst: List[String]): Boolean = {

    //First create a new list and keep track of the opening and closing brackets. Assign value "1" when "(", "-1" when ")" and 0 otherwise
    //Also make the first element 0, to make sure that we start from that point
    val transform  = 0 :: lst.map(el =>
      if(el.equals("(")) 1
      else if(el.equals(")")) -1
      else 0
    ).toList

    print(transform)

    //ReduceLeft is used as requested, and keep a partial sum of all the strings. In the end sum must be 0 to confirm that the parentheses are balanced
    val isBalanced = transform.reduceLeft( (x,y) => (x,y) match {
      case (0,-1) => 0
      case (x,-1) => x-1
      case (x,1) => x+1
      case (_,0) => x
      }
    )

    println(isBalanced)

    if(isBalanced == 0) true else false

  }


}
