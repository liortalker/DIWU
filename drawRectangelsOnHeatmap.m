
function fig = drawRectangelsOnHeatmap(heatmap, rectangels, lineWidth, colorVec, fig)

    if (~exist('fig'))
        fig = figure; imagesc(heatmap);
    else
        figure(fig); hold on;
    end
    numRect = size(rectangels,1);
    hold on;
    for i = 1:numRect
        rectangle('position',rectangels(i,:),'linewidth',lineWidth,'edgecolor',colorVec(i,:));
    end

end


