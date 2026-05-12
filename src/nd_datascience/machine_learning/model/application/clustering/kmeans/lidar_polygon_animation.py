from nd_sociomind.experiment.parts.oldest.uav1_normal_lidar_time_ranges_modality import \
    Uav1NormalLidarTimeRangesModality

import sys
import numpy

try:
    from PyQt6.QtCore import Qt, QTimer, QPointF
    from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
    from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSlider, QVBoxLayout, \
        QWidget

    QT_API_VERSION = 6
except ImportError:
    from PyQt5.QtCore import Qt, QTimer, QPointF
    from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
    from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSlider, QVBoxLayout, \
        QWidget

    QT_API_VERSION = 5


class QtCompatibility:
    def get_horizontal_orientation(self):
        if QT_API_VERSION == 6:
            return Qt.Orientation.Horizontal
        return Qt.Horizontal

    def get_space_key(self):
        if QT_API_VERSION == 6:
            return Qt.Key.Key_Space
        return Qt.Key_Space

    def get_right_key(self):
        if QT_API_VERSION == 6:
            return Qt.Key.Key_Right
        return Qt.Key_Right

    def get_left_key(self):
        if QT_API_VERSION == 6:
            return Qt.Key.Key_Left
        return Qt.Key_Left

    def get_escape_key(self):
        if QT_API_VERSION == 6:
            return Qt.Key.Key_Escape
        return Qt.Key_Escape

    def set_antialiasing(self, painter: QPainter) -> None:
        if QT_API_VERSION == 6:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        else:
            painter.setRenderHint(QPainter.Antialiasing, True)


class QtApplicationExecutor:
    def execute(self, application: QApplication) -> int:
        if hasattr(application, "exec"):
            return application.exec()
        return application.exec_()


class LidarPolygonDataLoader:
    def __init__(self, data_slice, lidar_dimension=720, start_angle=-numpy.pi, end_angle=numpy.pi,
                 far_range_threshold=5.0, replacement_distance=15.0):
        self.data_slice = data_slice
        self.lidar_dimension = lidar_dimension
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.far_range_threshold = far_range_threshold
        self.replacement_distance = replacement_distance

        self.time_ranges_modality = None
        self.time_ranges = None
        self.lidar_ranges_time_series = None
        self.x_time_series = None
        self.y_time_series = None
        self.angle_values = None

    def load(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        self.time_ranges_modality = Uav1NormalLidarTimeRangesModality(self.data_slice)
        self.time_ranges = self.time_ranges_modality.get_np_time_ranges()

        if self.time_ranges.ndim != 2:
            raise ValueError("LiDAR data must have shape (time_steps, features).")

        expected_column_count = self.lidar_dimension + 1

        if self.time_ranges.shape[1] != expected_column_count:
            raise ValueError(f"LiDAR data must have {expected_column_count} columns.")

        self.lidar_ranges_time_series = self.time_ranges[:, 1:]
        self.lidar_ranges_time_series = self._clean_ranges(self.lidar_ranges_time_series)
        self.x_time_series, self.y_time_series = self._convert_to_cartesian(self.lidar_ranges_time_series)

        return self.x_time_series, self.y_time_series

    def _clean_ranges(self, lidar_ranges_time_series: numpy.ndarray) -> numpy.ndarray:
        cleaned_ranges = lidar_ranges_time_series.copy()

        invalid_mask = ~numpy.isfinite(cleaned_ranges)
        far_mask = cleaned_ranges > self.far_range_threshold
        replacement_mask = numpy.logical_or(invalid_mask, far_mask)

        cleaned_ranges[replacement_mask] = self.replacement_distance

        negative_mask = cleaned_ranges < 0.0
        cleaned_ranges[negative_mask] = 0.0

        return cleaned_ranges

    def _convert_to_cartesian(self, lidar_ranges_time_series: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        angle_step = (self.end_angle - self.start_angle) / float(self.lidar_dimension)
        self.angle_values = self.start_angle + numpy.arange(self.lidar_dimension, dtype=float) * angle_step

        cosine_values = numpy.cos(self.angle_values)
        sine_values = numpy.sin(self.angle_values)

        x_time_series = lidar_ranges_time_series * cosine_values[None, :]
        y_time_series = lidar_ranges_time_series * sine_values[None, :]

        return x_time_series, y_time_series


class LidarPolygonCanvas(QWidget):
    def __init__(self, x_time_series: numpy.ndarray, y_time_series: numpy.ndarray, frame_stride=5,
                 display_axis_limit=16.0, parent=None):
        super().__init__(parent)

        self.qt_compatibility = QtCompatibility()

        self.x_time_series = x_time_series
        self.y_time_series = y_time_series
        self.frame_stride = frame_stride
        self.display_axis_limit = display_axis_limit

        self.current_frame_position = 0
        self.frame_indices = self._create_frame_indices()
        self.show_points = False
        self.fill_polygon = False

        self.background_color = QColor(255, 255, 255)
        self.grid_color = QColor(225, 225, 225)
        self.axis_color = QColor(180, 180, 180)
        self.line_color = QColor(30, 90, 180)
        self.point_color = QColor(180, 40, 40)
        self.fill_color = QColor(30, 90, 180, 35)

        self.axis_limit = self._calculate_axis_limit()

        self.setMinimumSize(700, 700)

    def set_frame_position(self, frame_position: int) -> None:
        if frame_position < 0:
            frame_position = 0

        maximum_position = len(self.frame_indices) - 1

        if frame_position > maximum_position:
            frame_position = maximum_position

        self.current_frame_position = frame_position
        self.update()

    def move_next(self) -> None:
        next_position = self.current_frame_position + 1

        if next_position >= len(self.frame_indices):
            next_position = 0

        self.current_frame_position = next_position
        self.update()

    def move_previous(self) -> None:
        previous_position = self.current_frame_position - 1

        if previous_position < 0:
            previous_position = len(self.frame_indices) - 1

        self.current_frame_position = previous_position
        self.update()

    def get_current_frame_index(self) -> int:
        return int(self.frame_indices[self.current_frame_position])

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        self.qt_compatibility.set_antialiasing(painter)

        painter.fillRect(self.rect(), self.background_color)

        self._draw_grid(painter)
        self._draw_axes(painter)
        self._draw_lidar_polygon(painter)
        self._draw_status_text(painter)

        painter.end()

    def _draw_grid(self, painter: QPainter) -> None:
        pen = QPen(self.grid_color)
        pen.setWidth(1)
        painter.setPen(pen)

        grid_step = self.axis_limit / 5.0

        for grid_index in range(-5, 6):
            value = float(grid_index) * grid_step

            start_vertical = self._world_to_screen(value, -self.axis_limit)
            end_vertical = self._world_to_screen(value, self.axis_limit)

            start_horizontal = self._world_to_screen(-self.axis_limit, value)
            end_horizontal = self._world_to_screen(self.axis_limit, value)

            painter.drawLine(start_vertical, end_vertical)
            painter.drawLine(start_horizontal, end_horizontal)

    def _draw_axes(self, painter: QPainter) -> None:
        pen = QPen(self.axis_color)
        pen.setWidth(2)
        painter.setPen(pen)

        horizontal_start = self._world_to_screen(-self.axis_limit, 0.0)
        horizontal_end = self._world_to_screen(self.axis_limit, 0.0)

        vertical_start = self._world_to_screen(0.0, -self.axis_limit)
        vertical_end = self._world_to_screen(0.0, self.axis_limit)

        painter.drawLine(horizontal_start, horizontal_end)
        painter.drawLine(vertical_start, vertical_end)

    def _draw_lidar_polygon(self, painter: QPainter) -> None:
        frame_index = self.get_current_frame_index()

        x_values = self.x_time_series[frame_index]
        y_values = self.y_time_series[frame_index]

        polygon = QPolygonF()

        for point_index in range(x_values.shape[0]):
            point = self._world_to_screen(float(x_values[point_index]), float(y_values[point_index]))
            polygon.append(point)

        pen = QPen(self.line_color)
        pen.setWidth(2)
        painter.setPen(pen)

        if self.fill_polygon:
            painter.setBrush(QBrush(self.fill_color))
        else:
            painter.setBrush(QBrush(QColor(0, 0, 0, 0)))

        painter.drawPolygon(polygon)

        if self.show_points:
            point_pen = QPen(self.point_color)
            point_pen.setWidth(4)
            painter.setPen(point_pen)

            for point_index in range(x_values.shape[0]):
                point = self._world_to_screen(float(x_values[point_index]), float(y_values[point_index]))
                painter.drawPoint(point)

    def _draw_status_text(self, painter: QPainter) -> None:
        frame_index = self.get_current_frame_index()
        text = f"Frame: {frame_index}    Position: {self.current_frame_position + 1}/{len(self.frame_indices)}"

        pen = QPen(QColor(20, 20, 20))
        painter.setPen(pen)
        painter.drawText(15, 25, text)

    def _world_to_screen(self, x_value: float, y_value: float) -> QPointF:
        width = max(self.width(), 1)
        height = max(self.height(), 1)

        margin = 30.0
        drawing_width = float(width) - 2.0 * margin
        drawing_height = float(height) - 2.0 * margin

        x_ratio = (x_value + self.axis_limit) / (2.0 * self.axis_limit)
        y_ratio = (y_value + self.axis_limit) / (2.0 * self.axis_limit)

        screen_x = margin + x_ratio * drawing_width
        screen_y = margin + (1.0 - y_ratio) * drawing_height

        return QPointF(screen_x, screen_y)

    def _create_frame_indices(self) -> numpy.ndarray:
        total_frame_count = self.x_time_series.shape[0]

        if self.frame_stride <= 1:
            frame_indices = numpy.arange(total_frame_count)
        else:
            frame_indices = numpy.arange(0, total_frame_count, self.frame_stride)

        return frame_indices

    def _calculate_axis_limit(self) -> float:
        if self.display_axis_limit is not None:
            return float(self.display_axis_limit)

        maximum_x = float(numpy.max(numpy.abs(self.x_time_series)))
        maximum_y = float(numpy.max(numpy.abs(self.y_time_series)))
        maximum_value = max(maximum_x, maximum_y, 1.0)

        axis_limit = maximum_value * 1.05

        return axis_limit


class LidarPolygonMainWindow(QMainWindow):
    def __init__(self, x_time_series: numpy.ndarray, y_time_series: numpy.ndarray, frame_stride=5,
                 interval_milliseconds=25, display_axis_limit=16.0):
        super().__init__()

        self.qt_compatibility = QtCompatibility()

        self.canvas = LidarPolygonCanvas(
            x_time_series=x_time_series,
            y_time_series=y_time_series,
            frame_stride=frame_stride,
            display_axis_limit=display_axis_limit
        )

        self.timer = QTimer(self)
        self.timer.setInterval(interval_milliseconds)
        self.timer.timeout.connect(self._advance_frame)

        self.play_button = QPushButton("Pause")
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.points_button = QPushButton("Points: Off")
        self.fill_button = QPushButton("Fill: Off")
        self.speed_label = QLabel(f"Interval: {interval_milliseconds} ms")
        self.frame_slider = QSlider(self.qt_compatibility.get_horizontal_orientation())

        self._build_user_interface()
        self._connect_events()

        self.timer.start()

    def keyPressEvent(self, event) -> None:
        key = event.key()

        if key == self.qt_compatibility.get_space_key():
            self._toggle_playback()
        elif key == self.qt_compatibility.get_right_key():
            self._step_next()
        elif key == self.qt_compatibility.get_left_key():
            self._step_previous()
        elif key == self.qt_compatibility.get_escape_key():
            self.close()
        else:
            super().keyPressEvent(event)

    def _build_user_interface(self) -> None:
        self.setWindowTitle("LiDAR Polygon Qt Player")

        central_widget = QWidget()
        main_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()

        maximum_slider_value = len(self.canvas.frame_indices) - 1
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(maximum_slider_value)
        self.frame_slider.setValue(0)

        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.previous_button)
        controls_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.points_button)
        controls_layout.addWidget(self.fill_button)
        controls_layout.addWidget(self.speed_label)

        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.frame_slider)
        main_layout.addLayout(controls_layout)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.resize(900, 950)

    def _connect_events(self) -> None:
        self.play_button.clicked.connect(self._toggle_playback)
        self.previous_button.clicked.connect(self._step_previous)
        self.next_button.clicked.connect(self._step_next)
        self.points_button.clicked.connect(self._toggle_points)
        self.fill_button.clicked.connect(self._toggle_fill)
        self.frame_slider.valueChanged.connect(self._slider_changed)

    def _advance_frame(self) -> None:
        self.canvas.move_next()
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.canvas.current_frame_position)
        self.frame_slider.blockSignals(False)

    def _toggle_playback(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.play_button.setText("Pause")

    def _step_next(self) -> None:
        self.canvas.move_next()
        self.frame_slider.setValue(self.canvas.current_frame_position)

    def _step_previous(self) -> None:
        self.canvas.move_previous()
        self.frame_slider.setValue(self.canvas.current_frame_position)

    def _toggle_points(self) -> None:
        self.canvas.show_points = not self.canvas.show_points

        if self.canvas.show_points:
            self.points_button.setText("Points: On")
        else:
            self.points_button.setText("Points: Off")

        self.canvas.update()

    def _toggle_fill(self) -> None:
        self.canvas.fill_polygon = not self.canvas.fill_polygon

        if self.canvas.fill_polygon:
            self.fill_button.setText("Fill: On")
        else:
            self.fill_button.setText("Fill: Off")

        self.canvas.update()

    def _slider_changed(self, value: int) -> None:
        self.canvas.set_frame_position(value)


class LidarPolygonQtApplication:
    def __init__(self, data_slice, lidar_dimension=720, start_angle=-numpy.pi, end_angle=numpy.pi,
                 far_range_threshold=5.0, replacement_distance=15.0, frame_stride=5, interval_milliseconds=25,
                 display_axis_limit=16.0):
        self.data_slice = data_slice
        self.lidar_dimension = lidar_dimension
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.far_range_threshold = far_range_threshold
        self.replacement_distance = replacement_distance
        self.frame_stride = frame_stride
        self.interval_milliseconds = interval_milliseconds
        self.display_axis_limit = display_axis_limit

    def run(self) -> None:
        data_loader = LidarPolygonDataLoader(
            data_slice=self.data_slice,
            lidar_dimension=self.lidar_dimension,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
            far_range_threshold=self.far_range_threshold,
            replacement_distance=self.replacement_distance
        )

        x_time_series, y_time_series = data_loader.load()

        application = QApplication(sys.argv)

        main_window = LidarPolygonMainWindow(
            x_time_series=x_time_series,
            y_time_series=y_time_series,
            frame_stride=self.frame_stride,
            interval_milliseconds=self.interval_milliseconds,
            display_axis_limit=self.display_axis_limit
        )

        main_window.show()

        application_executor = QtApplicationExecutor()
        exit_code = application_executor.execute(application)

        sys.exit(exit_code)


if __name__ == "__main__":
    qt_application = LidarPolygonQtApplication(
        data_slice=slice(0, 50000),
        lidar_dimension=720,
        start_angle=-numpy.pi,
        end_angle=numpy.pi,

        far_range_threshold=2.0,
        replacement_distance=4.0,

        frame_stride=5,
        interval_milliseconds=50,
        display_axis_limit=5.0
    )

    qt_application.run()