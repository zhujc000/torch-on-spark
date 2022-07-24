package in.z0.torchonspark.core;
import org.apache.commons.lang.StringUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TorchCommandRunner {

    private final String torchWapperScriptPath;
    private final int rank;
    private final int localRank;
    private final int numOfDevices;
    private final String mainIp;
    private final String workingDir;
    private final Map<String, String> other;
    private final String mainPort;
    private Process process;

    public TorchCommandRunner(String torchWapperScriptPath,
                              int rank,
                              int localRank,
                              int numOfDevices,
                              String mainIp,
                              String mainPort,
                              String workingDir,
                              Map<String, String> other){
        this.torchWapperScriptPath = torchWapperScriptPath;
        this.rank = rank;
        this.localRank = localRank;
        this.numOfDevices = numOfDevices;
        this.mainIp = mainIp;
        this.workingDir = workingDir;
        this.mainPort = mainPort;

        assert numOfDevices >= 1;
        assert rank >= 0 && rank < numOfDevices;
        assert localRank >=0 && localRank < numOfDevices;
        assert StringUtils.isNotEmpty(mainIp);
        assert StringUtils.isNotEmpty(workingDir);

        if(other == null || other.size() <= 0){
            this.other = new HashMap<String, String>();
        }else{
            this.other = other;
        }
    }

    private ProcessBuilder buildPb(){
        List<String> command = new ArrayList<String>();
        command.add("python");
        command.add("-u");
        command.add(torchWapperScriptPath);
        command.add("--rank");
        command.add(Integer.toString(rank));
        command.add("--local_rank");
        command.add(Integer.toString(localRank));
        command.add("--num_devices");
        command.add(Integer.toString(numOfDevices));
        command.add("--main_ip");
        command.add(mainIp);
        command.add("--main_port");
        command.add(mainPort);
        command.add("--working_dir");
        command.add(workingDir);

        ProcessBuilder pb = new ProcessBuilder();

        return pb;
    }

    public synchronized void exec() {
        if(process != null){
            return;
        }
        ProcessBuilder pb = buildPb();
        try {
            process = pb.start();
        }catch (Exception e){

        }
    }

    public synchronized Process getProcess(){
        return process;
    }

}
