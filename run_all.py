#!/usr/bin/env python

"""
Run the whole reconstruction pipeline.

Usage: `./run_all.py opensfm-dataset-dir`
"""

from collections import defaultdict
from copy import deepcopy
from functools import partial
from time import sleep
import numpy as np
import sys

from opensfm import align
from opensfm import dataset
from opensfm import features
from opensfm import matching
from opensfm import reconstruction
from opensfm import types

# Required to see debugging output from opensfm / ceres.
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


class Problem(object):
    """
    Encodes domain knowledge about the problem.
    """

    # Masks are image regions blacklisted for feature extraction.
    video2masks = {
        0: [
            dict(top=0.66, bottom=1.0, left=0.0, right=1.0),
            dict(top=0.0, bottom=0.2, left=0.0, right=0.25),
            dict(top=0.0, bottom=1.0, left=0.9, right=1.0),
        ],
        1: [
            dict(top=0.54, bottom=1.0, left=0.0, right=1.0),
            dict(top=0.0, bottom=1.0, left=0.9, right=1.0),
        ],
        2: [
            dict(top=0.68, bottom=1.0, left=0.0, right=1.0),
            dict(top=0.0, bottom=0.22, left=0.0, right=0.27),
            dict(top=0.0, bottom=1.0, left=0.88, right=1.0),
        ],
        3: [
            dict(top=0.61, bottom=1.0, left=0.0, right=1.0),
            dict(top=0.0, bottom=0.3, left=0.0, right=0.03),
            dict(top=0.0, bottom=1.0, left=0.9, right=1.0),
        ],
        4: [
            dict(top=0.66, bottom=1.0, left=0.0, right=1.0),
            dict(top=0.0, bottom=0.4, left=0.0, right=0.05),
            dict(top=0.0, bottom=1.0, left=0.88, right=1.0),
        ],
    }

    def __init__(self):
        # We have 5 videos of 10 frames each.
        self.videos = [0, 2, 3, 1, 4]
        self.frames = {v: range(10) for v in self.videos}

        self.image2masks = {}
        for v in self.videos:
            for f in self.frames[v]:
                self.image2masks[self.filename(v, f)] = self.video2masks[v]

        # Each video is shot with a potentially different camera.
        # This allows the optimization problem to choose different calibration
        # parameters (focal length, distortion params) for each camera.
        self.image2camera = {}
        for v in self.videos:
            camera = self.make_default_camera()
            camera.id = "camera_{}".format(v)
            for f in self.frames[v]:
                self.image2camera[self.filename(v, f)] = camera

    @staticmethod
    def make_default_camera():
        camera = types.PerspectiveCamera()
        camera.width, camera.height = 960, 720  # FIXME don't hardcode
        camera.focal_prior = camera.focal = 1.0
        camera.k1_prior = camera.k1 = 0.0
        camera.k2_prior = camera.k2 = 0.0
        return camera

    @staticmethod
    def filename(video_id, frame_id):
        return '{}_{}.jpg'.format(video_id, frame_id)

    @staticmethod
    def pair(video_id_1, frame_id_1, video_id_2, frame_id_2):
        return (
            Problem.filename(video_id_1, frame_id_1),
            Problem.filename(video_id_2, frame_id_2))

    @staticmethod
    def reco_triple(
            video_id_1, frame_id_1, video_id_2, frame_id_2, hint_forward):
        return (
            Problem.filename(video_id_1, frame_id_1),
            Problem.filename(video_id_2, frame_id_2),
            hint_forward)

    def get_pairs_to_match(self):
        """
        Return pairs of images in which to match features.
        """
        pairs = []
        # Adjacent frames in each video:
        for v in self.videos:
            for f1, f2 in zip(self.frames[v], self.frames[v][1:]):
                pairs.append(self.pair(v, f1, v, f2))
        # Corresponding frames in different videos:
        for i, v1 in enumerate(self.videos):
            for v2 in self.videos[i + 1:]:
                for f1, f2 in zip(self.frames[v1], self.frames[v2]):
                    pairs.append(self.pair(v1, f1, v2, f2))
        return pairs

    def get_reconstruction_order(self):
        """
        Return order in which to build reconstruction.

        Each element of the returned list is a `(image1, image2, hint_forward)`
        triple, where `image1` and `image2` are filenames, and hint_forward is
        a bool that indicates `image2` was shot from directly in front of
        `image1`. (We use this hint to make the reconstruction problem easier.)
        """
        order = []

        # Order 1: All frames from video 0, then all frames from video 1, etc.
        for i, v in enumerate(self.videos):
            if i > 0:
                prev_v = self.videos[i - 1]
                order.append(self.reco_triple(prev_v, 0, v, 0, False))
            for j, f in enumerate(self.frames[v]):
                if j > 0:
                    prev_f = self.frames[v][j - 1]
                    order.append(self.reco_triple(v, prev_f, v, f, True))

        # # Order 2: Frame 0 from all vids, then frame 1 from all vids, etc.
        # # This seems a little less robust than Order 1.
        # v1 = self.videos[0]
        # for v2 in self.videos[1:]:
        #     order.append(self.reco_triple(v1, 0, v2, 0, False))
        # for f1 in xrange(9):
        #     for v in self.videos:
        #         order.append(self.reco_triple(v, f1, v, f1 + 1, True))

        return order


def extract_features(problem, data):
    """
    Extract features from all images, and save to DataSet.
    """
    assert 'masks' not in data.config
    for image in data.images():
        if data.feature_index_exists(image):
            print "{} - extracting features (cached)".format(image)
        else:
            print "{} - extracting features".format(image)
            data.config['masks'] = problem.image2masks[image]
            points, descriptors, colors = features.extract_features(
                data.image_as_array(image), data.config)
            del data.config['masks']
            data.save_features(image, points, descriptors, colors)
            index = features.build_flann_index(descriptors, data.config)
            data.save_feature_index(image, index)


def get_candidate_matches(index1, descriptors2, config):
    """
    Helper for `match_custom`.

    For each feature in `descriptors2`, find its 2 nearest neighbors in
    `index1`, and return either the best one, or both, depending on how good
    they are relative to each other.

    Returns a list of `(feat1, feat2, distance)` triples.
    The words "distance" and "cost" are used interchangeably.
    """
    search_params = dict(checks=config.get('flann_checks', 200))
    squared_ratio = config.get('lowes_ratio', 0.6)**2
    results, dists = index1.knnSearch(descriptors2, 2, params=search_params)
    candidates = []
    for feat2 in xrange(len(descriptors2)):
        feat1a, feat1b = results[feat2, :]
        dist1a, dist1b = dists[feat2, :]
        if dist1a < squared_ratio * dist1b:
            # First match is much better than second match.
            candidates.append((feat1a, feat2, dist1a))
        else:
            # First match and second match are similarly good.
            candidates.append((feat1a, feat2, dist1a))
            candidates.append((feat1b, feat2, dist1b))
    return candidates


def match_custom(descriptors1, index1, descriptors2, index2, config):
    """
    Custom method for matching features in two images.

    (Note: Did not end up using this.)

    The idea is to be more forgiving than `matching.match_symmetric`, and
    return more candidate feature pairs. Relies on `get_candidate_matches` to
    get the initial matches, and then cleans up the resulting graph to make the
    matches symmetric. (Symmetric means that each feature in image1 matches at
    most 1 feature in image2, and vice versa.)
    """
    candidates_1_2 = get_candidate_matches(index1, descriptors2, config)
    candidates_2_1 = get_candidate_matches(index2, descriptors1, config)
    pair_to_cost = defaultdict(float)
    pair_to_num_seen = defaultdict(int)
    for feat1, feat2, cost in candidates_1_2:
        pair_to_cost[(feat1, feat2)] += cost
        pair_to_num_seen[(feat1, feat2)] += 1
    for feat2, feat1, cost in candidates_2_1:
        pair_to_cost[(feat1, feat2)] += cost
        pair_to_num_seen[(feat1, feat2)] += 1

    # Now we have a bipartite graph:
    # - left side nodes are features from image1
    # - right side nodes are features from image2
    # - on the edges we have the total cost, and num_seen (1 if the edge has
    #   been seen in one direction, or 2 if the edge has been seen in both
    #   directions)
    # Some nodes might have degree 2, and we want to cut some edges to make all
    # nodes have degree 0 or 1. Intuitively, if we have two edges from a node,
    # we want to cut the "worse" one (the one corresponding to a match that has
    # higher cost). This could be done optimally with min-cost max-flow, but we
    # opt for a fast greedy approach. First, for any node on the left side that
    # has 2 edges, we delete the worse edge. Then we repeat that for nodes on
    # the right side.

    matches = {}
    costs = {}
    for pair, cost in pair_to_cost.iteritems():
        if pair_to_num_seen[pair] != 2:
            continue
        if cost >= config.get('cberzan_max_cost', 1.0):
            continue
        feat1, feat2 = pair
        if (feat1 not in costs or cost < costs[feat1]):
            matches[feat1] = feat2
            costs[feat1] = cost
    back_matches = {}
    back_costs = {}
    for feat1, feat2 in matches.iteritems():
        cost = costs[feat1]
        if (feat2 not in back_costs or cost < back_costs[feat2]):
            back_matches[feat2] = feat1
            back_costs[feat2] = cost
    back_matches_arr = np.array(back_matches.items())
    matches_arr = back_matches_arr[:, [1, 0]]
    return matches_arr


def match_features(problem, data):
    """
    Match features between all relevant pairs of images, and save to DataSet.
    """
    pairs = problem.get_pairs_to_match()
    candidates = defaultdict(list)
    for image1, image2 in pairs:
        candidates[image1].append(image2)
    for image1, image2s in candidates.iteritems():
        if data.matches_exists(image1):
            print "{} - matching features (cached)".format(image1)
        else:
            print "{} - matching features".format(image1)
            points1, descriptors1, colors1 = data.load_features(image1)
            index1 = data.load_feature_index(image1, descriptors1)
            image1_matches = {}
            for image2 in image2s:
                print "{} - {} - matching features".format(image1, image2)
                points2, descriptors2, colors2 = data.load_features(image2)
                index2 = data.load_feature_index(image2, descriptors2)
                image1_matches[image2] = matching.match_symmetric(
                    descriptors1, index1, descriptors2, index2, data.config)
                # image1_matches[image2] = match_custom(
                #     descriptors1, index1, descriptors2, index2, data.config)
            data.save_matches(image1, image1_matches)


def create_tracks(problem, data):
    """
    Create tracks graph based on matches, and save to DataSet.

    A track is a feature that has been matched in 2 or more images. Each track
    has a unique id. In the reconstruction, every track becomes a point in 3D.

    The tracks graph is a bipartite graph with images on the left side and
    track ids on the right side. An edge indicates that the given track occurs
    in the given image. The edge stores info about that (image, track)
    correspondence.
    """
    try:
        data.load_tracks_graph()
        print "creating track graph (cached)"
        return
    except IOError:
        print "creating track graph"

    image2features = {}
    image2colors = {}
    for image in data.images():
        points, descriptors, colors = data.load_features(image)
        image2features[image] = points[:, :2]
        image2colors[image] = colors

    images2matches = {}
    for image1 in data.images():
        try:
            matches = data.load_matches(image1)
        except IOError:
            continue
        for image2 in matches:
            images2matches[image1, image2] = matches[image2]

    tracks_graph = matching.create_tracks_graph(
        image2features, image2colors, images2matches, data.config)
    data.save_tracks_graph(tracks_graph)


def get_empty_metadata():
    empty_metadata = types.ShotMetadata()
    empty_metadata.gps_position = [0.0, 0.0, 0.0]
    empty_metadata.gps_dop = 999999.0
    empty_metadata.orientation = 1
    return empty_metadata


def custom_two_view_reconstruction(
        p1, p2, camera1, camera2, thresh, hint_forward):
    """
    Find relative pose of `camera2` w.r.t. `camera1` based on matches.

    The matches are given as arrays of normalized 2D points, `p1` and `p2`.

    If `hint_forward` is True, retry until we find a transformation that places
    `camera2` in front of `camera`.

    Returns `R, t, inliers` where `R` is rotation as a Rodrigues vector, `t` is
    translation, and `inliers` is a list of indices in `p1` and `p2` that agree
    with this transformation. Note that the reconstruction is ambiguous up to
    scale, so the norm of `t` is meaningless.
    """
    max_trials = 100  # FIXME put const in config
    for trial in xrange(max_trials):
        R, t, inliers = reconstruction.two_view_reconstruction(
            p1, p2, camera1, camera2, thresh)
        t_norm = t / np.linalg.norm(t)
        # print "R={} t={} t_norm={} len(inliers)={}".format(
        #     R, t, t_norm, len(inliers))
        if hint_forward and t_norm[2] >= -0.75:  # FIXME put const in config
            print "hint_forward violated; retrying ({}/{})".format(
                trial + 1, max_trials)
            # HACK: Reconstruction.two_view_reconstruction uses
            # pyopengv.relative_pose_ransac, which initializes a
            # CentralRelativePoseSacProblem with the default options, including
            # randomSeed=true which means initializing the RNG with the current
            # time in seconds. This means that within the same second, we will
            # get exactly the same result. There is no way to avoid this while
            # using pyopengv, so we just sleep for a bit to get a new seed.
            sleep(0.5)
            continue
        break
    else:
        print "WARNING: hint_forward failed after {} trials".format(max_trials)
        # We failed to find a reconstruction that obeys hint_forward, so just
        # return an identity rotation and a straight-along-z-axis translation,
        # and hope that bundle adjustment will find a better solution.
        assert hint_forward
        R = np.zeros(3)
        t = np.array([0.0, 0.0, -1.0])
        inliers = "ignored"
    return R, t, inliers


def _debug_short(graph, reco, im1, im2, prefix):
    inlier_tracks = [track for track in graph[im2] if track in reco.points]
    print (
        "{}: reco has {} points; new shot {} has {} points in reco"
        .format(prefix, len(reco.points), im2, len(inlier_tracks)))


def _debug_verbose(graph, reco, im1, im2, prefix):
    print "{}:".format(prefix)

    inlier_tracks = [track for track in graph[im2] if track in reco.points]
    print "reco has {} points; new shot {} has {} points in reco".format(
        len(reco.points), im2, len(inlier_tracks))

    cooccur = {}
    for image in reco.shots:
        if image == im2:
            continue
        cooccurring_tracks = [
            track for track in inlier_tracks if track in graph[image]]
        cooccur[image] = len(cooccurring_tracks)
    print "tracks in reco and {} co-occur in {}".format(im2, cooccur)

    rel_pose = reco.shots[im2].pose.compose(reco.shots[im1].pose.inverse())
    print "new shot has rel pose R={} t={}".format(
        rel_pose.rotation, rel_pose.translation)

    print


def bootstrap_reconstruction(
        problem, data, graph, im1, im2, hint_forward=False):
    """
    Build 3D reconstruction based on two images `im1` and `im2`.

    See `custom_two_view_reconstruction` for the meaning of `hint_forward`.

    Returns the `Reconstruction` object, or None if reconstruction failed.
    """
    print "----------------"
    print "bootstrap_reconstruction({}, {}, hint_forward={})".format(
        im1, im2, hint_forward)

    camera1 = problem.image2camera[im1]
    camera2 = problem.image2camera[im2]
    cameras = {camera1.id: camera1, camera2.id: camera2}

    tracks, p1, p2 = matching.common_tracks(graph, im1, im2)
    print "Common tracks: {}".format(len(tracks))

    thresh = data.config.get('five_point_algo_threshold', 0.006)
    min_inliers = data.config.get('five_point_algo_min_inliers', 50)
    R, t, inliers = custom_two_view_reconstruction(
        p1, p2, camera1, camera2, thresh, hint_forward)
    print "bootstrap: R={} t={} len(inliers)={}".format(R, t, len(inliers))
    if len(inliers) <= 5:  # FIXME: put const in config
        print "bootstrap failed: not enough points in initial reconstruction"
        return

    # Reconstruction is up to scale; set translation to 1.
    # (This will be corrected later in the bundle adjustment step.)
    t /= np.linalg.norm(t)

    reco = types.Reconstruction()
    reco.cameras = cameras

    shot1 = types.Shot()
    shot1.id = im1
    shot1.camera = camera1
    shot1.pose = types.Pose()
    shot1.metadata = get_empty_metadata()
    reco.add_shot(shot1)

    shot2 = types.Shot()
    shot2.id = im2
    shot2.camera = camera2
    shot2.pose = types.Pose(R, t)
    shot2.metadata = get_empty_metadata()
    reco.add_shot(shot2)

    reconstruction.triangulate_shot_features(
        graph, reco, im1,
        data.config.get('triangulation_threshold', 0.004),
        data.config.get('triangulation_min_ray_angle', 2.0))
    if len(reco.points) < min_inliers:
        print "bootstrap failed: not enough points after triangulation"
        return

    reconstruction.bundle_single_view(graph, reco, im2, data.config)
    reconstruction.retriangulate(graph, reco, data.config)
    reconstruction.bundle_single_view(graph, reco, im2, data.config)

    debug = partial(_debug_short, graph, reco, im1, im2)
    debug("bootstraped reconstruction")

    return reco


def grow_reconstruction(
        problem, data, graph, reco, im1, im2, hint_forward=False):
    """
    Grow the given reconstruction `reco` by adding a new image `im2` to it.

    The new image `im2` is initially matched against an image `im1`, which must
    already exist in the reconstruction.

    See `custom_two_view_reconstruction` for the meaning of `hint_forward`.

    Updates the reconstruction in-place and returns `True` on success.
    """
    # FIXME:
    # - Make DRY with bootstrap_reconstruction; they look similar but they're
    #   different in subtle ways.
    # - Could probably align once at the end, instead of aligning at every
    #   step.
    # - Sometimes we get "Termination: NO_CONVERGENCE" from Ceres. Check for
    #   that error case and perhaps run it for more iterations.

    print "----------------"
    print "grow_reconstruction({}, {}, hint_forward={})".format(
        im1, im2, hint_forward)

    reconstruction.bundle(graph, reco, None, data.config)
    align.align_reconstruction(reco, None, data.config)

    assert im1 in reco.shots
    assert im2 not in reco.shots

    camera1 = problem.image2camera[im1]
    camera2 = problem.image2camera[im2]

    tracks, p1, p2 = matching.common_tracks(graph, im1, im2)
    print "Common tracks: {}".format(len(tracks))

    thresh = data.config.get('five_point_algo_threshold', 0.006)
    R, t, inliers = custom_two_view_reconstruction(
        p1, p2, camera1, camera2, thresh, hint_forward)
    print "grow: R={} t={} len(inliers)={}".format(R, t, len(inliers))
    if len(inliers) <= 5:
        print "grow failed: not enough points in initial reconstruction"
        return False

    # Reconstruction is up to scale; set translation to 1.
    # (This will be corrected later in the bundle adjustment step.)
    t /= np.linalg.norm(t)

    assert camera1.id in reco.cameras
    if camera2.id not in reco.cameras:
        reco.add_camera(camera2)

    shot1_pose = reco.shots[im1].pose
    shot2 = types.Shot()
    shot2.id = im2
    shot2.camera = camera2
    shot2.pose = types.Pose(R, t).compose(shot1_pose)
    shot2.metadata = get_empty_metadata()
    reco.add_shot(shot2)

    debug = partial(_debug_short, graph, reco, im1, im2)
    debug("started with")

    # FIXME: fail here if im2 does not have enough tracks in common with reco

    pose_before = deepcopy(reco.shots[im2].pose)
    reconstruction.bundle_single_view(graph, reco, im2, data.config)

    # It's possible that bundle_single_view caused hint_forward to be violated.
    # If that's the case, revert to the pose before bundle_single_view, and
    # hope that after triangulating the points from `im2`, bundle adjustment
    # will find a better solution.
    rel_pose_after = reco.shots[im2].pose.compose(
        reco.shots[im1].pose.inverse())
    t_after_norm = rel_pose_after.translation / np.linalg.norm(
        rel_pose_after.translation)
    # print "*** t_after_norm={}".format(t_after_norm)
    if hint_forward and t_after_norm[2] >= -0.75:  # FIXME put const in config
        print "*** hint_forward violated; undoing bundle_single_view"
        reco.shots[im2].pose = pose_before
        rel_pose_after = reco.shots[im2].pose.compose(
            reco.shots[im1].pose.inverse())
        t_after_norm = rel_pose_after.translation / np.linalg.norm(
            rel_pose_after.translation)
        print "*** after undo: t_after_norm={}".format(t_after_norm)
    # print

    reconstruction.triangulate_shot_features(
        graph, reco, im2,
        data.config.get('triangulation_threshold', 0.004),
        data.config.get('triangulation_min_ray_angle', 2.0))
    reconstruction.bundle(graph, reco, None, data.config)
    reconstruction.remove_outliers(graph, reco, data.config)
    align.align_reconstruction(reco, None, data.config)
    reconstruction.retriangulate(graph, reco, data.config)
    reconstruction.bundle(graph, reco, None, data.config)

    debug("ended with")

    return True


def reconstruct(problem, data):
    """
    Reconstruct 3D scene and save to the given DataSet.

    Relies on the feature matches and tracks graph to be already computed.

    Returns the reconstruction.
    """
    graph = data.load_tracks_graph()
    reco_triple = problem.get_reconstruction_order()
    reco = bootstrap_reconstruction(
        problem, data, graph,
        reco_triple[0][0], reco_triple[0][1], reco_triple[0][2])
    for prev_img, next_img, hint_forward in reco_triple[1:]:
        result = grow_reconstruction(
            problem, data, graph, reco, prev_img, next_img, hint_forward)
        if not result:
            print "Stopping because grow_reconstruction failed..."
            break

    # Give each 3D point in the reconstruction a color.
    reconstruction.paint_reconstruction(data, graph, reco)

    # Align the reconstruction with the ground plane for a sane visualization.
    align.align_reconstruction(reco, None, data.config)

    data.save_reconstruction([reco])
    return reco


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: {} dataset".format(sys.argv[0])
        sys.exit(1)
    problem = Problem()
    data = dataset.DataSet(sys.argv[1])

    extract_features(problem, data)
    match_features(problem, data)
    create_tracks(problem, data)
    reco = reconstruct(problem, data)
