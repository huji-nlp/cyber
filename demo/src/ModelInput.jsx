import React from 'react';
import Button from './components/Button'
import ModelIntro from './components/ModelIntro'


// These are some quickly-accessible examples to try out with your model.  They will get
// added to the select box on the demo page, and will auto-populate your input fields when they
// are selected.  The names here need to match what's read in `handleListChange` below.

const examples = [
  {
    label: "ebay example",
    text_input: "Due to the difference between different monitors, the picture may not reflect the actual color of the item.",
  },
  {
    label: "onion illegal example",
    text_input: "Product Price Quantity Pack of 10x1cc BD Insulin Syringes 10 USD 0.015 Bh. SAMPLER! One Point Of Heroin 4 (0.10g) 30 USD 0.046 Bh. GRAND OPENING SPECIAL QUATER GRAM HEROIN 4 (0.25g) 55 USD 0.084 Bh. HALF GRAM HEROIN 4 (0.50g) 100 USD 0.153 Bh. GRAND OPENING SPECIAL FULL WEIGHED GRAM HEROIN 4 (1.0g) 180 USD",
  },
  {
    label: "onion legal example",
    text_input: "Betnovate Betamethasone valerate Scalp application (lotion) 30 ml US$29.00 each, manufactured by: GlaxoSmithKline Identical to Beta-Val Betamethasone valerate Lotion 30 ml",
  }
];

// This determines what text shows up in the select box for each example.  The input to
// this function will be one of the items from the `examples` list above.
function summarizeExample(example) {
  return example.label;
}

// You can give a model name and description that show up in your demo.
const title = "Cyber Classification Demo";
const description = (
  <span>
  Classify a paragraph.
  </span>
);

class ModelInput extends React.Component {
  constructor(props) {
    super(props);
    this.handleListChange = this.handleListChange.bind(this);
    this.onClick = this.onClick.bind(this);
  }

  handleListChange(e) {
    if (e.target.value !== "") {
      // This gets called when the select box gets changed.  You want to set the values of
      // your input boxes with the content in your examples.
      this.text_input.value = examples[e.target.value].text_input
    }
  }

  onClick() {
    const { runModel } = this.props;

    // You need to map the values in your input boxes to json values that get sent to your
    // predictor.  The keys in this dictionary need to match what your predictor is expecting to receive.
    runModel({text_input: this.text_input.value});
  }

  render() {

    const { outputState } = this.props;

    return (
      <div className="model__content">
        <ModelIntro title={title} description={description} />
        <div className="form__instructions"><span>Enter text or</span>
          <select disabled={outputState === "working"} onChange={this.handleListChange}>
              <option value="">Choose an example...</option>
              {examples.map((example, index) => {
                return (
                    <option value={index} key={index}>{summarizeExample(example)}</option>
                );
              })}
          </select>
        </div>

       {/*
         * This is where you add your input fields.  You shouldn't have to change any of the
         * code in render() above here.  We're giving a couple of example inputs here, one for a
         * larger piece of text, like a paragraph (the `textarea`) and one for a shorter piece of
         * text, like a question (the `input`).  You'll probably want to change the variable names
         * here to match the input variable names in your model.
         */}

        <div className="form__field">
          <label>Text input</label>
          <textarea ref={(x) => this.text_input = x} type="text" autoFocus="true"></textarea>
        </div>
        {/*<div className="form__field">*/}
          {/*<label>Short text input</label>*/}
          {/*<input ref={(x) => this.short_text_input = x} type="text"/>*/}
        {/*</div>*/}

       {/* You also shouldn't have to change anything below here. */}

        <div className="form__field form__field--btn">
          <Button enabled={outputState !== "working"} onClick={this.onClick} />
        </div>
      </div>
    );
  }
}

export default ModelInput;
