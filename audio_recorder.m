% MATLAB code to record audio signal
for i=1:10
    dir = ['library\', mat2str(i-1)];
    mkdir(dir);
    for j=1:20
        recObj = audiorecorder(8000, 16, 2);
        a = [ '两秒内说', mat2str(i-1), '第', mat2str(j), '次'];
        disp(a)
        recordblocking(recObj, 2);
        disp('End of Recording.');
        myRecording = getaudiodata(recObj);
        filename = ['library\', mat2str(i-1), '\data', mat2str(j), '.wav'];
        audiowrite(filename, myRecording, 8000);
    end
end
