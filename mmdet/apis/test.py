import os.path as osp
# import os
import pickle
import random
import shutil
import tempfile

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.ops.minarearect import minaerarect
from mmcv.image import imread, imwrite
import cv2
from mmcv.visualization.color import Color, color_val
from mmdet.core import get_classes, tensor2imgs, rbbox2result
import matplotlib.pyplot as plt


def bt_pts_area(ps):
    b, n = ps.shape
    last_x = ps[:, -2]
    last_y = ps[:, -1]
    first_x = ps[:, 0]
    first_y = ps[:, 1]

    res = last_x * first_y - last_y * first_x
    for i in range(0, n - 2, 2):
        res += ps[:, i] * ps[:, i + 3] - ps[:, i + 1] * ps[:, i + 2]
    return res / 2.0


def single_d(p, d1, d2):
    [x1, y1, x2, y2, x3, y3] = p[:, 0], p[:, 1], d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1]

    s2 = (x1 * y2 - x1 * y3 + x2 * y3 - x2 * y1 + x3 * y1 - x3 * y2)
    a = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    d = np.abs(s2) / a
    return d


def mul_p(ps, rect_p):
    bt, len = rect_p.shape
    assert bt > 0
    assert len > 0
    total_temp = []
    for ind in range(0, 18, 2):
        p = ps[:, ind:ind + 2]
        temp = []
        for i in range(0, 6, 2):
            # 12, 23, 34
            temp.append(single_d(p, rect_p[:, i:i + 2], rect_p[:, i + 2:i + 4]))
        # 41
        temp.append(single_d(p, rect_p[:, 6:8], rect_p[:, 0:2]))
        # a = torch.tensor(temp)
        a = np.stack(temp, 0)
        # print(a.shape)
        a = np.min(a, 0)#.values
        # print(a.shape)
        total_temp.append(a)
    res = np.stack(total_temp, 0).mean(0)
    # print(res)
    # res = res.sum(dim=0)
    # print(res)
    return res
'''
__device__ inline double area(Point* ps,int n){
    ps[n]=ps[0];
    double res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}
'''
def normalized_sd(ps,rect_ps):
    ps_x = ps[0::2]
    ps_y = ps[1::2]
    area = np.abs(pts_area(rect_ps))
    edge = np.sqrt(area)
    # 保持量纲一致
    # nsd_x = np.var(ps_x) / np.sqrt(area)
    # nsd_y = np.var(ps_y) / np.sqrt(area)
    # 欧式距离
    d = (np.var(ps_y) + np.var(ps_x))/area
    nsd_x = np.std(ps_x) / edge
    nsd_y = np.std(ps_y) / edge
    return nsd_x, nsd_y, d

def pts_area(ps):
    n = len(ps)
    last_x = ps[-2]
    last_y = ps[-1]
    first_x = ps[0]
    first_y = ps[1]

    res = last_x*first_y - last_y*first_x
    for i in range(0, n -2, 2):
        res += ps[i] * ps[i+3] - ps[i+1]*ps[i+2]
    return res/2.0

def single_gpu_test(model, data_loader, show=False):
    load_results = True
    load_results = False
    # show = True
    show = False
    have_gui = False
    model.eval()
    results = []
    new_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    # for i, data in enumerate(data_loader):
    #     with torch.no_grad():
    #         result = model(return_loss=False, rescale=True, **data)
    #     results.append(result)
    # for i, data in enumerate(dataset):
    if load_results:
        import _pickle
        f = open('work_dirs/orientedreppoints_r50_demo/results.pkl', 'rb+')
        # f = open('results(author.pkl', 'rb+')
        results = _pickle.load(f)
    for i, data in enumerate(data_loader):
        if load_results:
        # data_loader 是随机排序的,上一次的顺序和这一次不同 所以可视化是乱的
        #     result = results[i]
            break
        # else:
        with torch.no_grad():
            # abc_list = np.array(data['img'][0])
            # img = torch.from_numpy(abc_list)
            #
            # abc = np.array(data['img'])
            # img = torch.from_numpy(abc)
            # img_t = img.permute(0, 3, 1, 2)
            # r = model.module.simple_test(img_t.cuda(), [data])
            # result( [0:18]:reppoints 9个点  [18:26]bbox 4个顶点   最后一个是score(阈值0.05) )
            # torch.cuda.empty_cache() #第二张还重新分配?
            # result = model(return_loss=False, rescale=True, img = [img_t], img_metas= [[data]])
            # 为什么只有第一张不需要重映射?因为rescale(即show)变量发生了变化
            # result_temp = model(return_loss=False, rescale=not show, **data)

            result = model(return_loss=False, rescale=True, **data)

            # 结果是以设置中的scale为基准,没有还原到各个图片shape上
            # dataloader将ori_shape转换到norm_scale, 现在从norm_scale映射到ori_shape, 默认保持长宽比例
            # ori_shape = data['img_metas'][0].data[0][0]['ori_shape']
            # img_shape = data['img_metas'][0].data[0][0]['img_shape']
            # scale_factor = data['img_metas'][0].data[0][0]['scale_factor']
            # result = result_temp.copy()
            # for item in result:
            #     item[:, 0:26] = item[:, 0:26] / scale_factor
            # print('result', result)
        # results.append(result)
        batch_size = data['img'][0].size(0)
        # print('result.len ',len(result))
        if batch_size >= 1:
            if result[0][0].shape[1] ==45:
                # init pro
                for bt in range(batch_size):
                    org_result = []
                    for item in result[bt]: # 15cls
                        bboxes = item[:, 36:45]
                        scatters = item[:, 0:18]
                        init_points = item[:, 18:36]
                        org_result.append(np.concatenate([scatters, bboxes], axis=1))
                    # print("result[bt].len",len(result[bt]))
                    # print("org_result.len",len(org_result))
                    new_results.append(result[bt])
                    results.append(org_result)
            elif result[0][0].shape[1] ==27:
                # demo
                for bt in range(batch_size):
                    results.append(result[bt])
            else:
                print("EORRO: unknow result[0][0].shape[1]")

        print(data['img_metas'][0].data[0][0]['filename'])
        if show and have_gui:
            pts = annotations[i]['bboxes']
            pts = pts.reshape((-1, 4, 2)).astype(int)
            # cv2.fillConvexPoly(img, pts, (255, 0, 0))
            # 可以一次性画多个, 所以可以跳出循环
            path = data['img_metas'][0].data[0][0]['filename']
            img = cv2.imread(path)
            # img = imread(path)
            img = np.ascontiguousarray(img)
            cv2.polylines(img, pts, isClosed=True, color=(0, 0, 255), thickness=2)
            plt.title("oriented GT")
            # plt.imshow(img)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

        if show:
            # model.module.show_result(data, result)
            # img_tensor = img_t
            # img_metas = [data]
            # imgs = tensor2imgs(img_tensor)
            # assert len(imgs) == len(img_metas)

            if dataset is None:
                class_names = model.module.CLASSES
            elif isinstance(dataset, str):
                class_names = get_classes(dataset)
            elif isinstance(dataset, (list, tuple)):
                class_names = dataset
            else:
                class_names = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
                 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
                 'swimming-pool', 'helicopter')
        

                # bbox_result, segm_result = result, None
                bbox_result, segm_result = result[0], None
                bboxes_result = np.vstack(bbox_result)

                # draw bounding boxes
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)
                # mmcv.imshow_det_bboxes(
                #     img_show,
                #     bboxes[:, 18:27], #前面18个是9个rep点
                #     labels,
                #     class_names=class_names,
                #     score_thr=0.5)

                show_init = bboxes_result.shape[1] == 9+18+18
                if bboxes_result.shape[1] == 9 + 18:
                    bboxes = bboxes_result[:, 18:27]
                    scatters = bboxes_result[:, 0:18]
                    init_points = None
                elif show_init:
                    bboxes = bboxes_result[:, 36:45]
                    scatters = bboxes_result[:, 0:18]
                    init_points = bboxes_result[:, 18:36]
                # scatters[:,1::2] =

                class_names = None
                # path = data['filename'][0]
                path = data['img_metas'][0].data[0][0]['filename']

                # img = data["img"][0]
                img = cv2.imread(path)
                # img = imread(path)
                img = np.ascontiguousarray(img)
                # score_thr = 0.7
                # score_thr = 0.5
                score_thr = 0.3
                # score_thr = 0.1
                if score_thr > 0.01:
                    assert bboxes.shape[1] == 5 or bboxes.shape[1] == 9
                    scores = bboxes[:, -1]
                    inds = scores > score_thr
                    bboxes = bboxes[inds, :]
                    scatters = scatters[inds, :]
                    labels = labels[inds]
                    if show_init:
                        init_points = init_points[inds, :]
                bbox_color = text_color = 'green'
                bbox_color = color_val(bbox_color)
                text_color = color_val(text_color)

                for bbox, label in zip(bboxes, labels):
                    bbox_int = bbox.astype(np.int32)
                    left_top = (bbox_int[0], bbox_int[1])
                    right_bottom = (bbox_int[2], bbox_int[3])
                    thickness = 1
                    if len(bbox) == 5 or len(bbox) == 4:
                        cv2.rectangle(
                            img, left_top, right_bottom, bbox_color, thickness=thickness)
                        label_text = class_names[
                            label] if class_names is not None else f'cls {label}'
                    if len(bbox) == 5:
                        label_text += f'|{bbox[-1]:.03f}'
                    if len(bbox) == 8 or len(bbox) == 9:
                        # 绘制未填充的多边形
                        # cv2.polylines(img, [bbox], isClosed=True, color=(0, 0, 255), thickness=1)

                        # 绘制填充的多边形
                        # pts = bbox[0:8]
                        # pts = pts.reshape((-1, 4, 2)).astype(int)
                        #
                        # triangle = np.array([[0, 0], [1000, 800], [0, 800]]).reshape((-1, 1, 2))
                        # cv2.fillConvexPoly(img, pts, (255, 0, 0))
                        # 可以一次性化多个
                        # cv2.polylines(img, pts, isClosed=True, color=(0, 0, 255), thickness=5)

                        # plt.title("dead v")
                        # plt.imshow(img)
                        # plt.show()
                        # cv2.fillPoly(img, triangle, color=(255, 255, 255))
                        label_text = class_names[
                            label] if class_names is not None else f'cls {label}'

                    if len(bbox) == 9:
                        label_text += f'|{bbox[-1]:.03f}'

                    # pts = bbox[0:8]
                    # pts = pts.reshape((-1, 4, 2)).astype(int)

                    # triangle = np.array([[0, 0], [1000, 800], [0, 800]]).reshape((-1, 1, 2))
                    # cv2.fillConvexPoly(img, pts, (255, 0, 0))
                    # 可以一次性画多个, 所以可以跳出循环
                    # cv2.polylines(img, pts, isClosed=True, color=(0, 0, 255), thickness=2)

                    font_scale = 0.5
                    cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                                cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
                pts = bboxes[:, 0:8]
                pts = pts.reshape((-1, 4, 2)).astype(int)
                # 绘制矩形框, 可以一次性画多个, 所以可以跳出循环
                cv2.polylines(img, pts, isClosed=True, color=(0, 0, 255), thickness=2)

                # 可以用上面画多边形的函数画散点图

                # 一个点的多边形就是散点
                scatters = scatters.reshape((-1, 1, 2)).astype(int)
                cv2.polylines(img, scatters, isClosed=True, color=(0, 255, 255), thickness=3)
                if show_init:
                    # init即细化前的自适应点
                    scatters_init = init_points.reshape((-1, 1, 2)).astype(int)
                    cv2.polylines(img, scatters_init, isClosed=True, color=(255, 0, 0), thickness=3)
                    # 将细化前后的两点连线
                    line = np.concatenate((scatters_init, scatters), axis=1)
                    cv2.polylines(img, line, isClosed=True, color=(0, 255, 0), thickness=1)

                win_name = ''
                wait_time = 0
                # TODO
                out_img = f"data/result_show/res_{i}.png" #None
                out_img = f"data/me/result_show/{osp.basename(path)}" #None

                # out_img = f"data/author/result_show/{osp.basename(path)}" #None
                
                if show and have_gui:
                    # cv2.imshow(win_name, imread(img))
                    plt.title("oriented v")
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # plt.imshow(img)
                    plt.show()
                    cv2.imshow(win_name, img)
                    if wait_time == 0:  # prevent from hanging if windows was closed
                        while True:
                            ret = cv2.waitKey(1)

                            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
                            # if user closed window or if some key pressed
                            if closed or ret != -1:
                                break
                    else:
                        ret = cv2.waitKey(wait_time)
                if out_img is not None:
                    # print("writing",out_img)
                    imwrite(img, out_img)
        if show:
            pass
            # img_metas = data
            # bbox_inputs = result + (img_metas, model.module.test_cfg, False)
            # bbox_list = model.module.bbox_head.get_bboxes(*bbox_inputs)
            # bbox_results = [
            #     rbbox2result(det_bboxes, det_labels, model.module.bbox_head.num_classes)
            #     for det_bboxes, det_labels in bbox_list
            # ]
            # wrap = {'img': [img_t], 'img_metas': [data]}
            # model.module.simple_test([img_t], [[data]])
            # model.module.show_result(data, r)
       
        
        # for item in result:
        #     bboxes = item [:, 36:45]
        #     scatters = item[:, 0:18]
        #     init_points = item[:, 18:36]
        #     org_result.append(np.concatenate([scatters, bboxes], axis=1))
        #     # new_result.append(np.concatenate([init_points, bboxes], axis=1))
        # # TODO 完成NSD改回来 end
        # new_results.append(result)
        # results.append(org_result)
        
        # batch_size = len(data['img']) 这个只是碰巧为1
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    # np.save(f"F:\\360downloads\OrientedRepPoints-main\\results.npy", results)
    # results = np.load(f"F:\\360downloads\OrientedRepPoints-main\\results.npy", allow_pickle=True)
    # if load_results:
    #     import _pickle
    #     f = open('F:\\360downloads\\OrientedRepPoints-main\\work_dirs\\orientedreppoints_r50_demo\\results.pkl', 'rb+')
    #     results = _pickle.load(f)
    '''
      bboxes = bboxes_result[:, 36:45]
      scatters = bboxes_result[:, 0:18]
      init_points = bboxes_result[:, 18:36]
      '''
    show_BD = True
    show_BD = False
    if show_BD:
        all_cls_res = np.concatenate([np.concatenate(b, 0) for b in results])
        temp1 = mul_p(all_cls_res[:, 0:18], all_cls_res[:, 18:26])
        area = bt_pts_area(all_cls_res[:, 18:26])
        border_dist = temp1 / (np.sqrt(np.abs(area)))
        print(f'\nborder_dist.mean():{border_dist.mean()} \n')

        all_cls_res = np.concatenate([np.concatenate(b, 0) for b in new_results])
        temp1 = mul_p(all_cls_res[:, 18:36], all_cls_res[:, 36:44])
        area = bt_pts_area(all_cls_res[:, 36:44])
        border_dist = temp1 / (np.sqrt(np.abs(area)))
        print(f'\ninit_stage_border_dist.mean():{border_dist.mean()} \n')
    show_NSD = True
    show_NSD = False
    if show_NSD:
        nsd_init_x_lis = []
        nsd_init_y_lis = []
        nsd_x_lis = []
        nsd_y_lis = []
        nsd_distace =[]
        nsd_distace_i= []
        print('new_results.len ',len(new_results))
        # print('new_results0.len ',len(new_results[0]))
        # print('new_results00.len ',len(new_results[0,0]))

        for result in new_results:
            for re in result:
                if re.shape[0] == 0:
                    continue
                for sample in re:
                    nsd_x, nsd_y,d = normalized_sd(sample[0:18], sample[-9:-1])
                    nsd_x_lis.append(nsd_x)
                    nsd_y_lis.append(nsd_y)
                    nsd_distace_i.append(d)
                    if True:
                        nsd_x_i, nsd_y_i,d = normalized_sd(sample[18:36], sample[-9:-1])
                        nsd_init_x_lis.append(nsd_x_i)
                        nsd_init_y_lis.append(nsd_y_i)
                        nsd_distace.append(d)
        print(
            f'\nmNSD_x:{np.mean(nsd_x_lis)},mNSD_y:{np.mean(nsd_y_lis)},mNSD_xi:{np.mean(nsd_init_x_lis)},mNSD_yi:{np.mean(nsd_init_y_lis)},mNSD_d:{np.mean(nsd_distace)},mNSD_di:{np.mean(nsd_distace_i)}')

    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
