import logging
from typing import Any, Dict, Optional

from pycaret.utils.generic import get_label_encoder

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def generate_classifier_explainer_dashboard(
        exp,
        estimator,
        display_format: str = "dash",
        dashboard_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,):

    """
        This function is changed from pycaret.classification.oop.dashboard()

        This function generates the interactive dashboard for a trained model.
        The dashboard is implemented using
        ExplainerDashboard (explainerdashboard.readthedocs.io)


        estimator: scikit-learn compatible object
            Trained model object


        display_format: str, default = 'dash'
            Render mode for the dashboard. The default is set to ``dash``
            which will
            render a dashboard in browser. There are four possible options:

            - 'dash' - displays the dashboard in browser
            - 'inline' - displays the dashboard in the jupyter notebook cell.
            - 'jupyterlab' - displays the dashboard in jupyterlab pane.
            - 'external' - displays the dashboard in a separate tab.
                (use in Colab)


        dashboard_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ``ExplainerDashboard`` class.


        run_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ``run``
            method of ``ExplainerDashboard``.


        **kwargs:
            Additional keyword arguments to pass to the ``ClassifierExplainer``
            or ``RegressionExplainer`` class.


        Returns:
            ExplainerDashboard
    """

    dashboard_kwargs = dashboard_kwargs or {}
    run_kwargs = run_kwargs or {}

    from explainerdashboard import ClassifierExplainer, ExplainerDashboard

    le = get_label_encoder(exp.pipeline)
    if le:
        labels_ = list(le.classes_)
    else:
        labels_ = None

    # Replaceing chars which dash doesnt accept for column name `.` , `{`, `}`

    X_test_df = exp.X_test_transformed.copy()
    LOG.info(X_test_df)
    X_test_df.columns = [
        col.replace(".", "__").replace("{", "__").replace("}", "__")
        for col in X_test_df.columns
    ]
    explainer = ClassifierExplainer(
        estimator, X_test_df, exp.y_test_transformed, labels=labels_, **kwargs
    )
    return ExplainerDashboard(
        explainer, mode=display_format,
        contributions=False, whatif=False,
        **dashboard_kwargs
    )


def generate_regression_explainer_dashboard(
        exp,
        estimator,
        display_format: str = "dash",
        dashboard_kwargs: Optional[Dict[str, Any]] = None,
        run_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,):

    """
    This function is changed from pycaret.regression.oop.dashboard()

        This function generates the interactive dashboard for a trained model.
        The dashboard is implemented using ExplainerDashboard
        (explainerdashboard.readthedocs.io)


        estimator: scikit-learn compatible object
            Trained model object


        display_format: str, default = 'dash'
            Render mode for the dashboard. The default is set to ``dash``
            which will
            render a dashboard in browser. There are four possible options:

            - 'dash' - displays the dashboard in browser
            - 'inline' - displays the dashboard in the jupyter notebook cell.
            - 'jupyterlab' - displays the dashboard in jupyterlab pane.
            - 'external' - displays the dashboard in a separate tab.
            (use in Colab)


        dashboard_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ``ExplainerDashboard`` class.


        run_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ``run`` method
            of ``ExplainerDashboard``.


        **kwargs:
            Additional keyword arguments to pass to the
            ``ClassifierExplainer`` or
            ``RegressionExplainer`` class.


        Returns:
            ExplainerDashboard
    """

    dashboard_kwargs = dashboard_kwargs or {}
    run_kwargs = run_kwargs or {}

    from explainerdashboard import ExplainerDashboard, RegressionExplainer

    # Replaceing chars which dash doesnt accept for column name `.` , `{`, `}`
    X_test_df = exp.X_test_transformed.copy()
    X_test_df.columns = [
        col.replace(".", "__").replace("{", "__").replace("}", "__")
        for col in X_test_df.columns
    ]
    explainer = RegressionExplainer(
        estimator, X_test_df, exp.y_test_transformed, **kwargs
    )
    return ExplainerDashboard(
        explainer,
        mode=display_format,
        contributions=False,
        whatif=False,
        shap_interaction=False,
        decision_trees=False,
        **dashboard_kwargs,
    )
