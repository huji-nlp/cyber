import React from 'react';
import HeatMap from './components/heatmap/HeatMap'
import Collapsible from 'react-collapsible'

class ModelOutput extends React.Component {
  render() {

    const { outputs } = this.props;

    // `outputs` will be the json dictionary returned by your predictor.  You can pull out
    // whatever you want here and visualize it.  We're giving some examples of different return
    // types you might have.  Change names for data types you want, and delete anything you don't
    // need.
    var label = outputs['label'];
    var all_labels = outputs['all_labels'];
    var class_probabilities = outputs['class_probabilities'].map(parseFloat);
    var prob = Math.max.apply(Math, class_probabilities).toFixed(5) * 100.0;
    // This is a 1D attention array, which we need to make into a 2D matrix to use with our heat
    // map component.
    var self_weights = outputs['self_weights'].map(x => [x]);
    // This is a 2D attention matrix.
    var attention_weights = outputs['attention_weights'];
    // Labels for our 2D attention matrix, and the rows in our 1D attention array.
    var tokens = outputs['tokens'];

    // This is how much horizontal space you'll get for the row labels.  Not great to have to
    // specify it like this, or with this name, but that's what we have right now.
    var xLabelWidth = "70px";

    return (
      <div className="model__content">

       {/*
         * This is where you display your output.  You can show whatever you want, however
         * you want.  We've got a few examples, of text-based output, and of visualizing model
         * internals using heat maps.
         */}

        <div className="form__field">
          <label>Label</label>
          <div className="model__content__summary">{ label } (p={ prob }%)</div>
        </div>

        <div className="form__field">
          <Collapsible trigger="Class probabilities">
            <table>
                <tr>{all_labels.map(x => <th>{x}</th>)}</tr>
                <tr>{class_probabilities.map(x => <td>{x}</td>)}</tr>
            </table>
          </Collapsible>
        </div>

        <div className="form__field">
          {/* We like using Collapsible to show model internals; you can keep this or change it. */}
          <Collapsible trigger="Model internals">
            <Collapsible trigger="1D attention">
                <HeatMap xLabels={['Attention']} yLabels={tokens} data={self_weights} xLabelWidth={xLabelWidth} />
            </Collapsible>
            <Collapsible trigger="2D attention">
                <HeatMap xLabels={tokens} yLabels={tokens} data={attention_weights} xLabelWidth={xLabelWidth} />
            </Collapsible>
          </Collapsible>
        </div>

      </div>
    );
  }
}

export default ModelOutput;
