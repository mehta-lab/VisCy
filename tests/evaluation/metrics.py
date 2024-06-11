"""
Borrowing functions from:
https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalInstanceLevelSemanticLabeling.py#L277 
CC BY-NC-SA 3.0 DEED License
"""


def evaluateMatches(matches, args):
    overlaps = args.overlaps
    min_region_sizes = [args.minRegionSizes[0]]
    dist_threshes = [args.distanceThs[0]]
    dist_confs = [args.distanceConfs[0]]

    # results: class x overlap
    ap = np.zeros(
        (len(dist_threshes), len(args.validClassLabels), len(overlaps)), np.float
    )
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
        zip(min_region_sizes, dist_threshes, dist_confs)
    ):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]["prediction"]:
                    for label_name in args.validClassLabels:
                        for p in matches[m]["prediction"][label_name]:
                            if "filename" in p:
                                pred_visited[p["filename"]] = False
            for li, label_name in enumerate(args.validClassLabels):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    predInstances = matches[m]["prediction"][label_name]
                    gtInstances = matches[m]["groundTruth"][label_name]
                    # filter groups in ground truth
                    gtInstances = [
                        gt
                        for gt in gtInstances
                        if gt["instance_id"] >= 1000
                        and gt["vert_count"] >= min_region_size
                        and gt["med_dist"] <= distance_thresh
                        and gt["dist_conf"] >= distance_conf
                    ]
                    if gtInstances:
                        has_gt = True
                    if predInstances:
                        has_pred = True

                    cur_true = np.ones(len(gtInstances))
                    cur_score = np.ones(len(gtInstances)) * (-float("inf"))
                    cur_match = np.zeros(len(gtInstances), dtype=np.bool)
                    # collect matches
                    for gti, gt in enumerate(gtInstances):
                        found_match = False
                        num_pred = len(gt["matched_pred"])
                        for pred in gt["matched_pred"]:
                            # greedy assignments
                            if pred_visited[pred["filename"]]:
                                continue
                            overlap = float(pred["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - pred["intersection"]
                            )
                            if overlap > overlap_th:
                                confidence = pred["confidence"]
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["filename"]] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in predInstances:
                        found_gt = False
                        for gt in pred["matchedGt"]:
                            overlap = float(gt["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - gt["intersection"]
                            )
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred["voidIntersection"]
                            for gt in pred["matchedGt"]:
                                # group?
                                if gt["instance_id"] < 1000:
                                    num_ignore += gt["intersection"]
                                # small ground truth instances
                                if (
                                    gt["vert_count"] < min_region_size
                                    or gt["med_dist"] > distance_thresh
                                    or gt["dist_conf"] < distance_conf
                                ):
                                    num_ignore += gt["intersection"]
                            proportion_ignore = float(num_ignore) / pred["vert_count"]
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(
                        y_score_sorted, return_index=True
                    )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    if len(y_true_sorted_cumsum) == 0:
                        num_true_examples = 0
                    else:
                        num_true_examples = y_true_sorted_cumsum[-1]
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.0)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float("nan")
                ap[di, li, oi] = ap_current
    return ap


def computeAverages(aps, args):
    dInf = 0
    o50 = np.where(np.isclose(args.overlaps, 0.5))
    o25 = np.where(np.isclose(args.overlaps, 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(args.overlaps, 0.25)))
    avg_dict = {}
    # avg_dict['allAp']     = np.nanmean(aps[ dInf,:,:  ])
    avg_dict["allAp"] = np.nanmean(aps[dInf, :, oAllBut25])
    avg_dict["allAp50%"] = np.nanmean(aps[dInf, :, o50])
    avg_dict["allAp25%"] = np.nanmean(aps[dInf, :, o25])
    avg_dict["classes"] = {}
    for li, label_name in enumerate(args.validClassLabels):
        avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ dInf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(aps[dInf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[dInf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[dInf, li, o25])
    return avg_dict
