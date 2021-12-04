class DriverAnimation:
    
    sx, sy, ex, ey = 7, 8, 9, 10
    st, et, vx, vy = 0, 1, 11, 12
    is_move, has_pass, f = 4,5,13
    
    def __init__(self, movement_matrix):
        self.movements = movement_matrix
        self.current_movement_index = 0

        self.move, self.hpass = False, False

        self.center = movement_matrix[0][self.sx], movement_matrix[0][self.sy]
        self.prev_center = self.center

        self.frame_count = 0

        self.time = 0
        
    def update(self):
        self.prev_center = self.center
        if self.current_movement_index < self.movements.shape[0]:
            m = self.movements[self.current_movement_index]
            if self.frame_count == 0:
                self.time = m[self.st]
                self.center = m[self.sx], m[self.sy]
                self.frame_count += 1
                self.move = m[self.is_move]
                self.hpass = m[self.has_pass]
            elif self.frame_count > m[self.f]:
                self.time = m[self.et]
                self.center = m[self.ex], m[self.ey]
                self.current_movement_index += 1
                self.frame_count = 0
            else:
                self.center = self.center[0] + m[self.vx], self.center[1] + m[self.vy]
                self.frame_count += 1
        return self.center
    
    #0 for not moving
    #1 for moving with passenger
    #2 for moving without passenger
    def state(self):
        if self.current_movement_index >= self.movements.shape[0]:
            return 0
        else:
            if self.hpass:
                return 1
            elif self.move:
                return 2
            else:
                return 0