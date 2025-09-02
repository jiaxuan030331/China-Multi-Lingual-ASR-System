sp_pid=`ps -ef | grep 'python3 run_server.py' | grep -v grep | awk '{print $2}'`

if [ -z "$sp_pid" ];

then

	 echo "[ not find server2 pid ]"

 else

	  echo "find result: $sp_pid "

	   kill -9 $sp_pid

   fi
