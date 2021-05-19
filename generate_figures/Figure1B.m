load('intermediate_mat_files\DataExtractedFromFigures.mat')

% make correction to glitch in Tm9
indChange=find(Tm9R>0.25);
indKeep = find(Tm9R<=0.25);
Tm9R = interp1(Tm9T(indKeep),Tm9R(indKeep),Tm9T);

% make changes to clear glitches in CT1Lobula
indLose = [76:78, 84,85, 90];
indKeep = setdiff([1:length(CT1LobulaT)],indLose);
CT1LobulaT = CT1LobulaT(indKeep);
CT1LobulaR = CT1LobulaR(indKeep);

%% now, plot up T4 inputs, normalized, with signs according to their inferred
% synapse onto T4

figure; hold on;
% plot(Mi1T,Mi1R,'k-');
plot(Mi9T,-Mi9R/max(abs(Mi9R)),'r-');
plot(Tm3T,Tm3R/max(Tm3R),'k-');
plot(Mi4T,-Mi4R/max(Mi4R),'b-');
plot(CT1MedullaT,-CT1MedullaR/max(CT1MedullaR),'m-');
plot([-3 0.5],[0 0],'k-');

legend('Mi9 (1)','Tm3 (2)','Mi4 (3)','CT1 (3)');

title('T4 inputs with signs, not including Mi1, which looks like Tm3');

%% now, do T5 similarly

figure; hold on;
plot(Tm9T,Tm9R,'r-');
plot(Tm1T,Tm1R,'k-');
plot(Tm2T,Tm2R,'k--');
plot(Tm4T,Tm4R,'k-.');
plot(CT1LobulaT,-CT1LobulaR/max(abs(CT1LobulaR)),'m-');
plot([-3 0.5],[0 0],'k-');

legend('Tm9 (1)','Tm1 (2)','Tm2 (2)','Tm4 (2)','CT1 (3)');

title('T5 inputs with signs');