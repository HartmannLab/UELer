# decortor for updating the status bar
def update_status_bar(func):
    def wrapper(*args, **kwargs):
        # Assume the first argument is self and that self.main_viewer is available
        self_obj = args[0]
        if hasattr(self_obj, "main_viewer"):
            viewer = getattr(self_obj, "main_viewer", None)
        # check if the first argument is the viewer itself
        else:
            viewer = self_obj        
        try:
            if viewer is not None:
                viewer.ui_component.status_bar.value = viewer._status_image["processing"]
                viewer.ui_component.status_bar.format = 'gif'
                viewer.ui_component.status_bar.width = 225
                viewer.ui_component.status_bar.height = 30
        except Exception as e:
            if viewer._debug:    
                print(f"Error updating status bar: {e}")

        try:
            func(*args, **kwargs)
        except AttributeError as e:
            print(f"Error: {e}")
            
        finally:
            try:
                if viewer is not None:
                    # Post-call status update
                    viewer.ui_component.status_bar.value = viewer._status_image["ready"]
                    viewer.ui_component.status_bar.format = 'png'
                    viewer.ui_component.status_bar.width = 30
                    viewer.ui_component.status_bar.height = 30
            except Exception as e:
                if viewer._debug:
                    print(f"Error updating status bar: {e}")    
    return wrapper