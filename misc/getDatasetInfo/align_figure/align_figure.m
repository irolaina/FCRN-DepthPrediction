function align_figure(hfig, target_mon, width, height, margin_top, margin_left, margin_btw_width, margin_btw_height)
%   align_figure: align figures without overlap
% 
%   Yonggyu Han / Yonsei university / 2016.02.18
% 
%   This function set figures' position to avoid overlap of figures.
%   
%   Example:
% 
%       hfig1 = figure();
%       hfig2 = figure();
%       hfig3 = figure();
%       hfig = [hfig1 hfig2 hfig3];
%       align_figure(hfig);
% 
%                                target_mon (for multiple monitor users)
%	ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
%	s
%	s                         margin_top
%	s
%	s                 wwwwwwwwwwwwwwwwwwwwwwwwww                      wwwwwwwwwwwwwwwwwwwwwwwwww
%	s                 w          width         w                      w                        w
%	s                 w                        w                      w                        w
%	s                 w h                      w                      w                        w
%	s                 w e                      w                      w                        w
%	s   margin_left   w i        fig1          w   margin_btw_width   w          fig2          w
%	s                 w g                      w                      w                        w
%	s                 w h                      w                      w                        w
%	s                 w t                      w                      w                        w
%	s                 w                        w                      w                        w
%	s                 w                        w                      w                        w
%	s                 wwwwwwwwwwwwwwwwwwwwwwwwww                      wwwwwwwwwwwwwwwwwwwwwwwwww
%	s
%	s                     margin_btw_height
%	s
%	s                 wwwwwwwwwwwwwwwwwwwwwwwwww                      wwwwwwwwwwwwwwwwwwwwwwwwww
%	s                 w                        w                      w                        w
%	s                 w                        w                      w                        w
%	s                 w                        w                      w                        w
%	s                 w                        w                      w                        w
%	s                 w          fig3          w                      w          fig4          w
%	s                 w                        w                      w                        w
%	s                 w                        w                      w                        w
%	s                 w                        w                      w                        w
%	s                 w                        w                      w                        w
%	s                 w                        w                      w                        w
%	s                 wwwwwwwwwwwwwwwwwwwwwwwwww                      wwwwwwwwwwwwwwwwwwwwwwwwww
%   s
%   s

    if nargin < 8
        if ~exist('margin_btw_height', 'var')
            margin_btw_height = 100;
        end   
        if nargin < 7
            if ~exist('margin_btw_width', 'var')
                margin_btw_width = 50;
            end   
            if nargin < 6
                if ~exist('margin_left', 'var')
                    margin_left = 30;
                end
                if nargin < 5
                    if ~exist('margin_top', 'var')
                        margin_top = 100;
                    end        
                    if nargin < 4
                        if ~exist('height', 'var')
                            pos_hfig_1 = get(hfig(1), 'Position'); % default size
                            height = pos_hfig_1(4);
                        end
                        if nargin < 3
                            if ~exist('width', 'var')
                                pos_hfig_1 = get(hfig(1), 'Position'); % default size
                                width = pos_hfig_1(3);
                            end 
                            if nargin < 2
                                if ~exist('target_mon', 'var')
                                    target_mon = 1;
                                end                                
                            end
                        end
                    end
                end
            end
        end
    end

    hmon = get(0,'MonitorPositions');
    offset_mon_left = hmon(target_mon, 1);
    offset_mon_top = hmon(target_mon, 2);    
    
    n_fig = length(hfig);    
    pos_screen = get(0,'ScreenSize');

    i = 0;
    j = 0;
    for k = 1:n_fig
        i = i + 1;
        if pos_screen(3)*(j+1) < ((i+1)*(width+margin_btw_width)+margin_left)
            j = j + 1;
            i = 1;
        end
        set(hfig(k), 'Position', [((i-1)*(width+margin_btw_width)+margin_left)+offset_mon_left (pos_screen(4)-height-margin_top -j*(height+margin_btw_height))-offset_mon_top  width height])
    end

end

