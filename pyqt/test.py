from PyQt5 import QtWidgets,uic
import json
class set_config_result_print():
    def set_step():
        source = call.input_folder_box.text()
        resize_true = call.true_resize.isChecked()
        resize_false = call.false_resize.isChecked()
        save_config_path_box = call.save_config_path_box.text()
        save_config_path_name = call.save_config_path_name.text()
        process_list = []
        for process_name in range(1, 16):
            exec(f'step_{process_name} = call.step{process_name}_box.currentText()')
            exec(f'if "-- Select Step" not in step_{process_name}: process_list.append(step_{process_name})')
        print("source:", source)
        if (resize_true == "False"):
            resize_status = resize_false
        else:
            resize_status = resize_true
        print("resize:", resize_status)
        print("process:", process_list)
        print("config_path:", save_config_path_box)
        print("config_name_format:", save_config_path_name)

        with open('data.json', 'r+') as f:
            data = json.load(f)
            data['id'] = 134  # <--- add `id` value.
            f.seek(0)  # <--- should reset file position to the beginning.
            json.dump(data, f, indent=4)
            f.truncate()  # remove remaining part

app=QtWidgets.QApplication([])
call=uic.loadUi("test.ui")

call.confirm.clicked.connect(set_config_result_print.set_step)

call.show()
app.exec()