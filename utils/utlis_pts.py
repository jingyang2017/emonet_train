import numpy as np
from skimage.draw import line

def LineDrawer_51(image, shape):
    groups = [
            np.arange(0, 5, 1),  # left eyebrow
            np.arange(5, 10,1),  # right eyebrow
            np.arange(10, 14, 1),  # nose
            np.arange(14, 19, 1),  # nosestrills
            np.arange(19, 25, 1),  # left eye
            np.arange(25, 31, 1),  # right eye
            np.arange(31, 43, 1),  # outer lips
            np.arange(43,51,1), # inner lips
        ]

    for g in groups:
        for i in range(len(g) - 1):
            start = shape[g[i]]
            end = shape[g[i + 1]]
            rr, cc = line(start[1], start[0], end[1], end[0])
            try:
                image[rr, cc, 0] = 255
                image[rr, cc, 1] = 0
                image[rr, cc, 2] = 0
            except:
                pass
    return image

def get_pts(heatmaps):
    heatmaps = heatmaps.squeeze().cpu().numpy()
    ans = []
    for i in range(heatmaps.shape[0]):
        tmp = heatmaps[i, ...]
        x, y = np.unravel_index(np.argmax(tmp), tmp.shape)
        xx, yy = x, y
        x, y = x+0.5, y+0.5

        if tmp[xx, min(yy+1, tmp.shape[1]-1)]>tmp[xx, max(yy-1,0)]:
            y+=0.25
        else:
            y-=0.25

        if tmp[min(xx+1, tmp.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
            x += 0.25
        else:
            x -= 0.25

        x, y = np.array([y,x])
        ans.append((x,y))
    ans = np.array(ans)
    return ans