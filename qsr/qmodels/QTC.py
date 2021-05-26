import time
import os
import json
import math
from numpy import linalg as LA
import numpy as np
def get_vertical_line(x1,y1,x2,y2):
	a = 2*(x2-x1)
	b = 2*(y2-y1)
	c = -(a*x2 + b*y2)
	if a<=0 and b<=0 and c<=0:
		return abs(a), abs(b), abs(c)
	else:
		return a, b, c
	
def get_qtc_feature(df):
	"""
	get the QTC relations between before and after state
	:param df: dataframe
	:return: QTC relation list, e,g,(+,-), + indicates a moves away from b, - means the distance between a and b is reducing
	"""
	flag_approach = []
	start_a = df[0][0:2]
	start_b = df[0][2:4]
	end_a = df[1][0:2]
	end_b = df[1][2:4]
	A1, B1, C1 = get_vertical_line(start_b[0], start_b[1], start_a[0], start_a[1])
	A2, B2, C2 = get_vertical_line(start_a[0], start_a[1], start_b[0], start_b[1])
	D1 = A1*end_a[0]+B1*end_a[1]+C1
	D2 = A2*end_b[0]+B2*end_b[1]+C2

	if start_a[0] < start_b[0]:
		if D1 <= 0:
		     flag_approach.append("+")
		elif D1 > 0:
		     flag_approach.append("-")
		# else:
		# 	flag_approach.append("0")
		if D2 < 0:
		     flag_approach.append("-")
		elif D2 >= 0:
		     flag_approach.append("+")
		# else:
		# 	flag_approach.append("0")
	elif start_a[0] > start_b[0]:
		if D1 < 0:
		     flag_approach.append("-")
		elif D1 >= 0:
		     flag_approach.append("+")
		# else:
		# 	flag_approach.append("0")
		if D2 <= 0:
		     flag_approach.append("+")
		elif D2 > 0:
		     flag_approach.append("-")
		# else:
		# 	flag_approach.append("0")
	else:
		flag_approach.append("+")
		flag_approach.append("+")

	dist_start = math.sqrt(math.pow((start_b[0] - start_a[0]), 2) + math.pow((start_b[1] - start_a[1]), 2))
	dist_end = math.sqrt(math.pow((end_b[0] - end_a[0]), 2) + math.pow((end_b[1] - end_a[1]), 2))
	if abs(start_a[0] - end_a[0])<0.5 and abs(start_a[1] - end_a[1])<0.5:
		flag_approach[0] = "0"
	if abs(start_b[0] - end_b[0])<0.5 and abs(start_b[1] - end_b[1])<0.5:
		flag_approach[1] = "0"
	return flag_approach
def QTCb_pq(df):
    """
    QTC-B (QTC Basic) represents the 1D relative motion of these two points.
    It uses a 2-tuple of qualitative relations (t1,t2),
    where each element can assume any of the values {-, 0, +} as follows:
    -   t1 movement of p with respect to q:
            [-] p is moving towards q
            [0] p is stable with respect to q
            [+] p is moving away from q
        t1 can be represented by the sign of the cosine of the angle between P vector and QP vector, using dot product.
            P  is the vector formed by points p and pn
            QP is with q and p.
    -   t2 movement of q with respect to p: as above, but swapping p and q
    """
    p = np.array(df[0][0:2])
    q = np.array(df[0][2:4])
    pn = np.array(df[1][0:2])
    qn = np.array(df[1][2:4])
    proj_tol = 0.001
    norm_tol = 0.001

    if (p.ndim != pn.ndim) or (p.ndim != q.ndim) or (p.ndim > 1):
        print("Not all elements have dimension 1")
        return
    elif (p.shape != pn.shape) or (p.shape != q.shape):
        print("Not all elements have same num of components")
        return

    # vector pointing next position of p: P
    P = pn - p
    modP = LA.norm(P)

    # vector pointing next position of p: P
    Q = qn - q
    modQ = LA.norm(Q)

    # vector between p and reference q: QP
    QP = q - p

    # and oposite
    PQ = p - q

    # |PQ| == |QP|
    modQP = LA.norm(QP)

    # dot product of P and PQ
    dotP = P.dot(PQ)

    # dot product of Q and QP
    dotQ = Q.dot(QP)

    if modQP:
        # projection of vector P over QP vector
        p_over_q = dotP / modQP

        # projection of vector Q over PQ vector
        q_over_p = dotQ / modQP

#         # normal to vector P and QP vectors
#         p_normal_q = crosP / modQP

#         # normal to vector Q and PQ vectors
#         q_normal_p = crosQ / modQP
    else:
        # p and q are the same point ...
        p_over_q = 0
        q_over_p = 0

    t1 = getSign(p_over_q, proj_tol)
    t2 = getSign(q_over_p, proj_tol)

    return [t1, t2]

def getSign(val, tol):
    if (val > tol):
        signo = '+'
    elif (val < -tol):
        signo = '-'
    else:
        signo = '0'
    return signo

start_time = time.time()
data_dir = "/home/richie/Desktop/pddl/geojson/gt_batch_new_split_eval_gt_level_2_type_2_604_naive/"
all_data = os.listdir(data_dir)
for level in all_data:
	if int(level) in range(0,150):
		index_list = []
		one_level = os.listdir(data_dir+level)
		for js in one_level:
			if "_center.json" in js:
				index_list.append(int(js.split("_")[0]))
		start = min(index_list)
		end = max(index_list)
		for ind in range(end, start, -1):
			print("start", ind)
			with open(data_dir + level + "/" + "{}_center.json".format(ind)) as cur_cent:
				curcent_json = json.load(cur_cent)
			with open(data_dir + level + "/" + "{}_center.json".format(ind-1)) as pre_cent:
				precent_json = json.load(pre_cent)
	
			for key, value in curcent_json.items():
				if key in precent_json.keys():
					curcent_json[key] = QTCb_pq([precent_json[key], value])
				else:
					curcent_json[key] = QTCb_pq([value, value])
			# os.remove(data_dir + level + "/" + "{}_center.json".format(ind))
			with open(data_dir + level + "/" + "{}_qtc.json".format(ind), 'w') as fp:
				fp.write(json.dumps(curcent_json, sort_keys=True))
		with open(data_dir + level + "/" + "{}_center.json".format(start)) as start_cent:
				stacent_json = json.load(start_cent)
		for key, value in stacent_json.items():
				stacent_json[key] = get_qtc_feature([value, value])
		# os.remove(data_dir + level + "/" + "{}_center.json".format(start))
		with open(data_dir + level + "/" + "{}_qtc.json".format(start), 'w') as fp:
			fp.write(json.dumps(stacent_json, sort_keys=True))


