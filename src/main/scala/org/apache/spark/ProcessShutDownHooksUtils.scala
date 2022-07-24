package org.apache.spark

import org.apache.spark.util.ShutdownHookManager

import java.util

object ProcessShutDownHooksUtils {

  private val runningProcessQueue = new util.LinkedList[Process]()
  ShutdownHookManager.addShutdownHook(() => {
    while(!runningProcessQueue.isEmpty){
      val process: Process = runningProcessQueue.removeLast()
      process.destroy()
    }
  })

  def addProcessKillHook(process: Process): Unit = this.synchronized {
    runningProcessQueue.add(process)
  }

}
