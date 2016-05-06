function logo = imseqlogo(pwm, width, height, rna)
    [num_nt, num_seq] = size(pwm);
    letterwidth = floor(width/num_seq);

    % load alphaget
    Aimg = dlmread('A.txt');
    Aimg = reshape(Aimg, [72, 65, 3]);
    Cimg = dlmread('C.txt');
    Cimg = reshape(Cimg, [76, 64, 3]);
    Gimg = dlmread('G.txt');
    Gimg = reshape(Gimg, [76, 67, 3]);
    if rna == 1
        Timg = dlmread('U.txt');
        Timg = reshape(Timg, [74, 57, 3]);
    else
        Timg = dlmread('T.txt');
        Timg = reshape(Timg, [72, 59, 3]);
    end
    logo = uint8(ones(height, width, 3)) * 255;

    % Calcuate the height matrix, based on log2 entropy
    heights = zeros(num_nt, num_seq);
    for i = 1:num_seq
        totheight = (2 - entropy(pwm(:, i)))*height/2;
        %totheight = height
        heights(:,i) = floor(pwm(:,i)*totheight);
    end

    % Just go through resizing and pasting in the glyphs
    for i = 1:num_seq
        y = height;
        [letterheight, index] = sort(heights(:, i));
        for j = 1:length(index)
            if(letterheight(j) == 0)
                continue;
            end
            switch(index(j))
                case 1
                    ntimg = imresize(Aimg, [letterheight(j) letterwidth]);
                case 2
                    ntimg = imresize(Cimg, [letterheight(j) letterwidth]);
                case 3
                    ntimg = imresize(Gimg, [letterheight(j) letterwidth]);
                case 4
                    ntimg = imresize(Timg, [letterheight(j) letterwidth]);
            end
            
            logo(y-letterheight(j)+1:y, (i-1)*letterwidth+1:(i-1)*letterwidth+letterwidth,:) = ntimg;
            y = y - letterheight(j);
        end
    end
end


% Calcuate entropy of a sequence
% p - probability distribution (assume sums to unity)
function s = entropy(p)
  s = 0;
  for ii = 1 : length(p)
      if(p(ii) > 0)
          s = s - p(ii) * log(p(ii));
      end
  end
end
            
