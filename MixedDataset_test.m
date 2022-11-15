clc
close all
clear all
N_L=20000;
Dataset(N_L,N_L,N_L,N_L);
function []=Dataset(N_2psk,N_4psk,N_8psk,N_16qam)
    L=128;
    n=0:L-1;
    n = n';
    for snr=-10:2:10
    name=['test/snr=',num2str(snr),'.mat'];
    IQ_2psk=zeros(N_2psk,L,2);
    for i=1:N_2psk
        x=randi([0,1],L,1);
        y_com = pskmod(x,2);
        y_com = y_com./std(y_com);
        y_com = y_com.*exp(1i*rand()*pi/16+1i*2*pi*0.1*n);
        y_noise = awgn(y_com, snr, 'measured');
        x = real(y_noise);
        y = imag(y_noise);
        IQ_2psk(i,:,1)=x;
        IQ_2psk(i,:,2)=y;
    end

    IQ_4psk=zeros(N_4psk,L,2);
    for i=1:N_4psk
        x=randi([0,3],L,1);
        y_com=pskmod(x,4);
        y_com = y_com./std(y_com);
        y_com = y_com.*exp(1i*rand()*pi/16+1i*2*pi*0.1*n);
        y_noise = awgn(y_com, snr, 'measured');
        x = real(y_noise);
        y = imag(y_noise);
        IQ_4psk(i,:,1)=x;
        IQ_4psk(i,:,2)=y;
    end

    IQ_8psk=zeros(N_8psk,L,2);
    for i=1:N_8psk
        x=randi([0,7],L,1);
        y_com=pskmod(x,8);
        y_com = y_com./std(y_com);
        y_com = y_com.*exp(1i*rand()*pi/16+1i*2*pi*0.1*n);
        y_noise = awgn(y_com, snr, 'measured');
        x = real(y_noise);
        y = imag(y_noise);
        IQ_8psk(i,:,1)=x;
        IQ_8psk(i,:,2)=y;
    end

    %16qam
    IQ_16qam=zeros(N_16qam,L,2);
    for i=1:N_16qam
        x=randi([0,15],L,1);
        y_com=qammod(x,16);
        y_com = y_com./std(y_com);
        y_com = y_com.*exp(1i*rand()*pi/16+1i*2*pi*0.1*n);
        y_noise = awgn(y_com, snr, 'measured');
        x = real(y_noise);
        y = imag(y_noise);
        IQ_16qam(i,:,1)=x;
        IQ_16qam(i,:,2)=y;
    end

    IQ=cat(1,IQ_2psk,IQ_4psk,IQ_8psk,IQ_16qam);
    save(name,'IQ')  
    end
end