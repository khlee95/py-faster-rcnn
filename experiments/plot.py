import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

def main():
    fig = plt.figure()
    ax1=fig.add_subplot(221)
    ax2=fig.add_subplot(222)
    ax3=fig.add_subplot(223)
    ax4=fig.add_subplot(224)
            
    #filename=sys.argv[1]
    filename='./parsed_log/faster_rcnn_end2end_ZF_.txt.2017-01-13_21-15-13.train'
    bbox_loss=[]
    cls_loss=[]
    rpn_cls_loss=[]
    rpn_loss_bbox=[]
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            inx_bbox_loss = row.index('bbox_loss')
            inx_cls_loss = row.index('cls_loss')
            inx_rpn_cls_loss = row.index('rpn_cls_loss')
            inx_rpn_loss_bbox = row.index('rpn_loss_bbox')
            break;
        for row in reader:
            bbox_loss.append(row[inx_bbox_loss])
            cls_loss.append(row[inx_cls_loss])
            rpn_cls_loss.append(row[inx_rpn_cls_loss])    
            rpn_loss_bbox.append(row[inx_rpn_loss_bbox])
    size_arr = 3499
    num_addition = (size_arr)/50+1
    avg_bbox_loss=[]
    avg_cls_loss=[]
    avg_rpn_cls_loss=[]
    avg_rpn_loss_bbox=[]
    cur_val=0;
    for i in range(50):
        avg1=0.0
        avg2=0.0
        avg3=0.0
        avg4=0.0
        final=0
        for j in range(num_addition):
            avg1=avg1+float(bbox_loss[num_addition*i+j])
            avg2=avg2+float(cls_loss[num_addition*i+j])
            avg3=avg3+float(rpn_cls_loss[num_addition*i+j])
            avg4=avg4+float(rpn_loss_bbox[num_addition*i+j])
            final=j+1
            if num_addition*i+j==3498:
                break;              
        avg1=avg1/final
        avg2=avg2/final
        avg3=avg3/final
        avg4=avg4/final
        avg_bbox_loss.append(avg1)
        avg_cls_loss.append(avg2)
        avg_rpn_cls_loss.append(avg3)
        avg_rpn_loss_bbox.append(avg4)
    f.close()
    iteration = np.linspace(1400, 70000, 50)
    ax1.plot(iteration, avg_bbox_loss , 'r.-')
    ax2.plot(iteration, avg_cls_loss , 'r.-')
    ax3.plot(iteration, avg_rpn_cls_loss , 'r.-')
    ax4.plot(iteration, avg_rpn_loss_bbox , 'r.-')
    plt.show()  



if __name__=='__main__':
    main()

