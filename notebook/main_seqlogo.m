


pwm = rand([4,12]);
pwm = pwm./repmat(sum(pwm),4,1);

bpwidth = 30;
height = 300;
num_seq = size(pwm,2);
width = bpwidth*num_seq;

rna=1;
logo = imseqlogo(pwm, width, height, rna);

letterwidth = floor(width/num_seq);
matrix = zeros(4*letterwidth, width, 3);

num_grads = 20;
cm = hsv(num_grads);
MAX = 1;
pwm(pwm>MAX) = MAX;
grads = 0:1/(num_grads-1):MAX;
pwm3 = zeros(4, num_seq, 3);
for i = 1:num_seq
    for j = 1:4
        [MIN, index] = min((grads-pwm(j,i)).^2);
        index
        pwm3(j,i,:) = cm(index,:)*255;
    end
end

for i = 1:num_seq
    index1 = i*letterwidth - letterwidth + 1: i*letterwidth;
    for j = 1:4
        index2 = j*letterwidth - letterwidth + 1: j*letterwidth;
        for k = 1:3
            matrix(index2, index1, k) = pwm3(j,i,k);
        end
    end
end

buffer = 3;
img = ones(height+4*letterwidth + buffer, width, 3)*255;
img(1:height,:,:) = logo;
img(end-4*letterwidth+1:end,:,:) = matrix;

figure; imshow(img);


%%
list = [];
for i = 1:num_seq
    for j = 1:4
        value = pwm(j,i);
        [MIN, index] = min((value-grads).^2);
        list = [list; index];
    end
end

unique(list)



