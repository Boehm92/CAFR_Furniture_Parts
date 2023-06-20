import os
import math
import numpy as np
import numpy.random
import madcad as mdc


class MachiningFeature:
    def __init__(self, machining_feature, wooden_board, board_length, board_height):
        self.machining_feature = machining_feature
        self.wooden_board = wooden_board
        self.board_length = board_length
        self.board_height = board_height
        self.max_machining_feature_dimension = 18

        self.machining_feature_functions = [self.o_ring, self.trough_hole,
                                            self.blind_hole, self.triangular_passage,
                                            self.rectangular_passage, self.circular_trough_slot,
                                            self.rectangular_trough_slot, self.rectangular_blind_slot,
                                            self.triangular_pocket, self.rectangular_pocket,
                                            self.circular_end_pocket, self.six_side_passage, self.six_side_pocket,
                                            self.chamfer]

    def apply_feature(self):
        return self.machining_feature_functions[self.machining_feature](self.wooden_board, self.board_length,
                                                                        self.board_height)

    def o_ring(self, wooden_board, board_length, board_height):

        _outside_ring_radius = np.random.uniform(1, self.max_machining_feature_dimension)
        _inside_ring_radius = np.random.uniform(_outside_ring_radius / 3, _outside_ring_radius - 0.2)
        _position_x = np.random.uniform(_outside_ring_radius + 0.5, board_length - _outside_ring_radius)
        _position_z = np.random.uniform(_outside_ring_radius + 0.5, board_height - _outside_ring_radius)
        _depth = np.random.uniform(1, 9)

        outside_ring = mdc.cylinder(
            mdc.vec3(_position_x, _depth, _position_z), mdc.vec3(_position_x, 25.01, _position_z),
            _outside_ring_radius)
        inside_ring = mdc.cylinder(
            mdc.vec3(_position_x, _depth, _position_z), mdc.vec3(_position_x, 25.01, _position_z),
            _inside_ring_radius)
        o_ring = mdc.difference(outside_ring, inside_ring)

        updated_model = mdc.difference(wooden_board, o_ring)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def trough_hole(self, wooden_board, board_length, board_height):

        _radius = np.random.uniform(0.5, self.max_machining_feature_dimension)
        _position_x = np.random.uniform(_radius + 0.5, board_length - _radius)
        _position_z = np.random.uniform(_radius + 0.5, board_height - _radius)

        cylinder = mdc.cylinder(
            mdc.vec3(_position_x, -0.1, _position_z), mdc.vec3(_position_x, 20.01, _position_z), _radius)

        updated_model = mdc.difference(wooden_board, cylinder)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def blind_hole(self, wooden_board, board_length, board_height):

        _radius = np.random.uniform(0.5, self.max_machining_feature_dimension)
        _position_x = np.random.uniform(_radius + 0.5, board_length - _radius)
        _position_z = np.random.uniform(_radius + 0.5, board_height - _radius)
        _depth = np.random.uniform(1, 17)

        cylinder = mdc.cylinder(
            mdc.vec3(_position_x, _depth, _position_z), mdc.vec3(_position_x, 18.01, _position_z), _radius)

        updated_model = mdc.difference(wooden_board, cylinder)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def triangular_passage(self, wooden_board, board_length, board_height):

        L = np.random.uniform(1, self.max_machining_feature_dimension)
        D = 18.02
        X = np.random.uniform(0.5, board_length - L)
        Z = np.random.uniform(0.5, board_height - L)
        A = mdc.vec3(0, 18.01, 0)
        B = mdc.vec3(0, 18.01, L)
        C = mdc.vec3(L * math.sin(math.radians(60)), 18.01, L / 2)

        triangular_passage = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, A)]
        triangular_passage = mdc.extrusion(-D * mdc.Y, mdc.flatsurface(triangular_passage))
        triangular_passage = triangular_passage.transform(mdc.translate(mdc.vec3(X, 0, Z)))

        updated_model = mdc.difference(wooden_board, triangular_passage)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def rectangular_passage(self, wooden_board, board_length, board_height):

        L = np.random.uniform(1, self.max_machining_feature_dimension)
        W = np.random.uniform(1, self.max_machining_feature_dimension)
        Depth = 18.02
        X = np.random.uniform(0.5, board_length - (L + 0.5))
        Z = np.random.uniform(0.5, board_height - (W + 0.5))
        A = mdc.vec3(0, 18.01, 0)
        B = mdc.vec3(0, 18.01, W)
        C = mdc.vec3(L, 18.01, W)
        D = mdc.vec3(L, 18.01, 0)

        _rectangular_passage = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_passage = mdc.extrusion(-Depth * mdc.Y, mdc.flatsurface(_rectangular_passage))
        _rectangular_passage = _rectangular_passage.transform(mdc.translate(mdc.vec3(X, 0, Z)))

        updated_model = mdc.difference(wooden_board, _rectangular_passage)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def circular_trough_slot(self, wooden_board, board_length, board_height):

        _radius = np.random.uniform(1, (self.max_machining_feature_dimension / 2))
        X = np.random.uniform(_radius + 0.5, board_length - _radius)

        _circular_trough_slot = mdc.cylinder(mdc.vec3(X, -0.01, board_height),
                                             mdc.vec3(X, 18.01, board_height), _radius)

        updated_model = mdc.difference(wooden_board, _circular_trough_slot)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def rectangular_trough_slot(self, wooden_board, board_length, board_height):

        W = np.random.uniform(1, self.max_machining_feature_dimension)
        D = np.random.uniform(1, self.max_machining_feature_dimension)
        H = 18.02
        X = np.random.uniform(0.5, board_length - W)
        A = mdc.vec3(0, -0.01, (board_height + 0.01))
        B = mdc.vec3(W, -0.01, (board_height + 0.01))
        C = mdc.vec3(W, -0.01, board_height - D)
        D = mdc.vec3(0, -0.01, board_height - D)

        _rectangular_trough_slot = [mdc.Segment(B, A), mdc.Segment(C, B), mdc.Segment(D, C), mdc.Segment(A, D)]
        _rectangular_trough_slot = mdc.extrusion(H * mdc.Y, mdc.flatsurface(_rectangular_trough_slot))
        _rectangular_trough_slot = _rectangular_trough_slot.transform(mdc.translate(mdc.vec3(X, 0, 0)))

        updated_model = mdc.difference(wooden_board, _rectangular_trough_slot)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def rectangular_blind_slot(self, wooden_board, board_length, board_height):
        W = np.random.uniform(1, self.max_machining_feature_dimension)
        D = np.random.uniform(1, self.max_machining_feature_dimension)
        H = np.random.uniform(1, 17)
        X = np.random.uniform(0.5, board_length - W)
        A = mdc.vec3(0, -0.01, (board_height + 0.01))
        B = mdc.vec3(W, -0.01, (board_height + 0.01))
        C = mdc.vec3(W, -0.01, board_height - D)
        D = mdc.vec3(0, -0.01, board_height - D)

        _rectangular_blind_slot = [mdc.Segment(B, A), mdc.Segment(C, B), mdc.Segment(D, C), mdc.Segment(A, D)]
        _rectangular_blind_slot = mdc.extrusion(H * mdc.Y, mdc.flatsurface(_rectangular_blind_slot))
        _rectangular_blind_slot = _rectangular_blind_slot.transform(mdc.translate(mdc.vec3(X, 0, 0)))

        updated_model = mdc.difference(wooden_board, _rectangular_blind_slot)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def triangular_pocket(self, wooden_board, board_length, board_height):

        L = np.random.uniform(1, self.max_machining_feature_dimension)
        D = np.random.uniform(1, 17)
        X = np.random.uniform(0.5, board_length - L)
        Z = np.random.uniform(0.5, board_height - L)
        A = mdc.vec3(0, -0.01, 0)
        B = mdc.vec3(L, -0.01, 0)
        C = mdc.vec3(L / 2, -0.01, L * math.sin(math.radians(60)))

        _triangular_pocket = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, A)]
        _triangular_pocket = mdc.extrusion(D * mdc.Y, mdc.flatsurface(_triangular_pocket))
        _triangular_pocket = _triangular_pocket.transform(mdc.translate(mdc.vec3(X, 0, Z)))

        updated_model = mdc.difference(wooden_board, _triangular_pocket)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def rectangular_pocket(self, wooden_board, board_length, board_height):

        L = np.random.uniform(1, self.max_machining_feature_dimension)
        W = np.random.uniform(1, self.max_machining_feature_dimension)
        Depth = np.random.uniform(1, 17)
        X = np.random.uniform(0.5, board_length - (L + 0.5))
        Z = np.random.uniform(0.5, board_height - (W + 0.5))
        A = mdc.vec3(0, -0.01, 0)
        B = mdc.vec3(W, -0.01, 0)
        C = mdc.vec3(W, -0.01, L)
        D = mdc.vec3(0, -0.01, L)

        _rectangular_pocket = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_pocket = mdc.extrusion(Depth * mdc.Y, mdc.flatsurface(_rectangular_pocket))
        _rectangular_pocket = _rectangular_pocket.transform(mdc.translate(mdc.vec3(X, 0, Z)))

        updated_model = mdc.difference(wooden_board, _rectangular_pocket)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def circular_end_pocket(self, wooden_board, board_length, board_height):

        _width = np.random.uniform(1, self.max_machining_feature_dimension - 2)
        _length = np.random.uniform(1, self.max_machining_feature_dimension - _width)
        _depth = np.random.uniform(1, 17)
        X = np.random.uniform(0.5, board_length - _width)
        Z = np.random.uniform(0.5, board_height - (_width + _length))

        A = mdc.vec3(0, -0.01, (_width / 2))
        B = mdc.vec3((_width / 2), -0.01, 0)
        C = mdc.vec3(_width, -0.01, (_width / 2))
        D = mdc.vec3(_width, -0.01, (_length + (_width / 2)))
        E = mdc.vec3( (_width / 2), -0.01, (_length + _width))
        F = mdc.vec3( 0, -0.01, (_length + (_width / 2)))

        _circular_end_pocket = [mdc.ArcThrough(A, B, C), mdc.Segment(C, D), mdc.ArcThrough(D, E, F), mdc.Segment(F, A)]
        _circular_end_pocket = mdc.extrusion(_depth * mdc.Y, mdc.flatsurface(_circular_end_pocket))
        _circular_end_pocket = _circular_end_pocket.transform(mdc.translate(mdc.vec3(X, 0, Z)))

        updated_model = mdc.difference(wooden_board, _circular_end_pocket)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def six_side_passage(self, wooden_board, board_length, board_height):

        _radius = np.random.uniform(1, (self.max_machining_feature_dimension / 2))
        _Cx = np.random.uniform(_radius + 0.5, board_length - _radius)
        _Cz = np.random.uniform(_radius + 0.5, board_height - _radius)
        _depth = 18.02

        A = mdc.vec3(_radius, -0.01, 0)
        B = mdc.vec3(_radius * math.cos(math.radians(-300)), -0.01, _radius * math.sin(math.radians(-300)))
        C = mdc.vec3(_radius * math.cos(math.radians(-240)), -0.01, _radius * math.sin(math.radians(-240)))
        D = mdc.vec3(- _radius, -0.01, 0)
        E = mdc.vec3(_radius * math.cos(math.radians(-120)), -0.01,_radius * math.sin(math.radians(-120)))
        F = mdc.vec3(_radius * math.cos(math.radians(-60)), -0.01, _radius * math.sin(math.radians(-60)))

        _six_side_passage = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, E),
                             mdc.Segment(E, F), mdc.Segment(F, A)]
        _six_side_passage = mdc.extrusion(_depth * mdc.Y, mdc.flatsurface(_six_side_passage))
        _six_side_passage = _six_side_passage.transform(mdc.translate(mdc.vec3(_Cx, 0, _Cz)))

        updated_model = mdc.difference(wooden_board, _six_side_passage)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def six_side_pocket(self, wooden_board, board_length, board_height):

        _radius = np.random.uniform(1, (self.max_machining_feature_dimension / 2))
        _Cx = np.random.uniform(_radius + 0.5, board_length - _radius)
        _Cz = np.random.uniform(_radius + 0.5, board_height - _radius)
        _depth = np.random.uniform(1, 17)

        A = mdc.vec3(_radius, -0.01, 0)
        B = mdc.vec3(_radius * math.cos(math.radians(-300)), -0.01, _radius * math.sin(math.radians(-300)))
        C = mdc.vec3(_radius * math.cos(math.radians(-240)), -0.01, _radius * math.sin(math.radians(-240)))
        D = mdc.vec3(- _radius, -0.01, 0)
        E = mdc.vec3(_radius * math.cos(math.radians(-120)), -0.01,_radius * math.sin(math.radians(-120)))
        F = mdc.vec3(_radius * math.cos(math.radians(-60)), -0.01, _radius * math.sin(math.radians(-60)))

        _six_side_pocket = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, E),
                            mdc.Segment(E, F), mdc.Segment(F, A)]
        _six_side_pocket = mdc.extrusion(_depth * mdc.Y, mdc.flatsurface(_six_side_pocket))
        _six_side_pocket = _six_side_pocket.transform(mdc.translate(mdc.vec3(_Cx, 0, _Cz)))

        updated_model = mdc.difference(wooden_board, _six_side_pocket)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def chamfer(self, wooden_board, board_length, board_height):

        edges = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 5), (0, 4)]
        selected_edge = np.random.randint(0, 6)
        chanmfer = [edges[selected_edge]]
        mdc.chamfer(wooden_board, chanmfer, ("width", 5))

        updated_model = wooden_board
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model